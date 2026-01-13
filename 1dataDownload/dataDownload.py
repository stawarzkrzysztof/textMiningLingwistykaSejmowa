#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sejm API bulk downloader (term-based) for text-mining.

Outputs (default under ./data/raw/term{TERM} and ./data/tables/term{TERM}):

data/raw/term10/
  mp/
    mp_list.json
    details/{mp_id}.json
  clubs/
    clubs.json
  proceedings/
    proceedings.json
    details/{proceedingNum}.json
  transcripts/
    index_by_day/p{proceedingNum}/d{YYYY-MM-DD}.json
    html_by_mp/{mp_id}-{slug}/p{proceedingNum}/d{YYYY-MM-DD}/s{statementNum}.html
    html_non_mp/p{proceedingNum}/d{YYYY-MM-DD}/s{statementNum}.html
  interpellations/
    pages/offset=..._limit=....json
    by_num/{num}/details.json
    by_num/{num}/body.html
    by_num/{num}/replies/{key}.html
    by_num/{num}/attachments/{key}/{filename}
    by_author_ptr/{mp_id}-{slug}/i{num}.json
  writtenQuestions/
    pages/...
    by_num/{num}/details.json
    by_num/{num}/body.html
    by_num/{num}/replies/{key}.html
    by_num/{num}/attachments/{key}/{filename}
    by_author_ptr/{mp_id}-{slug}/q{num}.json
  committees/ (optional)
    committees.json
    sittings/{code}_sittings.json
    sittings/{code}/{num}/sitting.html
    sittings/{code}/{num}/sitting.pdf

data/tables/term10/
  dim_mp.csv
  dim_club.csv
  fact_transcript_statement_index.csv
  ptr_interpellations_by_author.csv
  ptr_writtenQuestions_by_author.csv

Notes:
- No preprocessing: texts are stored as HTML/PDF/JSON.
- Resume-friendly: if file exists and non-empty, it is skipped (unless --force).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from tqdm import tqdm


# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("sejm")


# -----------------------------
# Helpers
# -----------------------------
_slug_re = re.compile(r"[^A-Za-z0-9]+")


def slugify(s: str, max_len: int = 80) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _slug_re.sub("-", s).strip("-").lower()
    return s[:max_len] if s else "unknown"


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if not s:
            return None
        if re.fullmatch(r"\d+", s):
            return int(s)
        return None
    except Exception:
        return None


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _backoff(attempt: int, base: float = 0.6, cap: float = 12.0) -> float:
    t = min(cap, base * (2**attempt))
    return t * (0.6 + random.random() * 0.8)


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    status_code: int
    url: str
    path: Optional[Path] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class Config:
    term: int
    api_root: str
    user_agent: str

    project_root: Path
    data_dir: Path
    raw_dir: Path
    tables_dir: Path
    logs_dir: Path

    http2: bool
    concurrency: int
    max_connections: int
    max_keepalive: int
    page_limit: int

    force: bool

    download_transcripts: bool
    download_interpellations: bool
    download_written_questions: bool
    download_attachments: bool
    download_committees: bool


def ensure_dirs(cfg: Config) -> None:
    # MP
    (cfg.raw_dir / "mp").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "mp" / "details").mkdir(parents=True, exist_ok=True)

    # Clubs
    (cfg.raw_dir / "clubs").mkdir(parents=True, exist_ok=True)

    # Proceedings + transcripts
    (cfg.raw_dir / "proceedings").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "proceedings" / "details").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "transcripts" / "index_by_day").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "transcripts" / "html_by_mp").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "transcripts" / "html_non_mp").mkdir(parents=True, exist_ok=True)

    # Interpellations
    (cfg.raw_dir / "interpellations" / "pages").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "interpellations" / "by_num").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "interpellations" / "by_author_ptr").mkdir(
        parents=True, exist_ok=True
    )

    # Written questions
    (cfg.raw_dir / "writtenQuestions" / "pages").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "writtenQuestions" / "by_num").mkdir(parents=True, exist_ok=True)
    (cfg.raw_dir / "writtenQuestions" / "by_author_ptr").mkdir(
        parents=True, exist_ok=True
    )

    # Committees
    (cfg.raw_dir / "committees" / "sittings").mkdir(parents=True, exist_ok=True)

    # Tables + logs
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)


def make_client(cfg: Config) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=cfg.max_connections, max_keepalive_connections=cfg.max_keepalive
    )
    headers = {"User-Agent": cfg.user_agent}
    return httpx.AsyncClient(
        http2=cfg.http2, headers=headers, limits=limits, follow_redirects=True
    )


async def request_with_retries(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    timeout_s: float = 30.0,
    max_retries: int = 7,
    accept_404: bool = False,
) -> httpx.Response:
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            r = await client.request(
                method, url, headers=headers, params=params, timeout=timeout_s
            )

            if accept_404 and r.status_code == 404:
                return r

            # Retry only on transient codes
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_retries:
                    r.raise_for_status()
                await asyncio.sleep(_backoff(attempt))
                continue

            r.raise_for_status()
            return r

        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPStatusError,
        ) as e:
            last_err = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(_backoff(attempt))

    raise last_err if last_err else RuntimeError("request failed")


async def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, text, encoding="utf-8")


async def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_bytes, data)


async def write_json(path: Path, obj: Any) -> None:
    await write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


# -----------------------------
# Generic worker pool (bounded memory, fast on large job counts)
# -----------------------------
async def run_job_pool(
    cfg: Config,
    jobs: list[Any],
    handler,  # async (client, job) -> FetchResult
    desc: str,
) -> tuple[int, int]:
    if not jobs:
        return 0, 0

    q: asyncio.Queue[Any] = asyncio.Queue()
    for job in jobs:
        q.put_nowait(job)
    for _ in range(cfg.concurrency):
        q.put_nowait(None)

    ok = 0
    fail = 0
    lock = asyncio.Lock()

    async with make_client(cfg) as client:
        pbar = tqdm(total=len(jobs), desc=desc)

        async def worker() -> None:
            nonlocal ok, fail
            while True:
                job = await q.get()
                try:
                    if job is None:
                        return
                    res: FetchResult = await handler(client, job)
                    async with lock:
                        if res.ok:
                            ok += 1
                        else:
                            fail += 1
                        pbar.update(1)
                finally:
                    q.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(cfg.concurrency)]
        await q.join()
        for t in tasks:
            t.cancel()
        pbar.close()

    return ok, fail


# -----------------------------
# Fetch primitives (with cache / force)
# -----------------------------
async def fetch_json_to_file(
    cfg: Config,
    client: httpx.AsyncClient,
    url: str,
    path: Path,
    *,
    accept_404: bool = False,
) -> FetchResult:
    if not cfg.force and path.exists() and path.stat().st_size > 0:
        return FetchResult(ok=True, status_code=200, url=url, path=path)

    try:
        r = await request_with_retries(
            client,
            "GET",
            url,
            headers={"Accept": "application/json"},
            accept_404=accept_404,
        )
        if accept_404 and r.status_code == 404:
            await write_text(path, "")  # marker
            return FetchResult(
                ok=False, status_code=404, url=url, path=path, error="404"
            )
        await write_json(path, r.json())
        return FetchResult(ok=True, status_code=r.status_code, url=url, path=path)
    except Exception as e:
        logger.exception("JSON fetch failed: %s", url)
        return FetchResult(ok=False, status_code=0, url=url, path=path, error=str(e))


async def fetch_text_to_file(
    cfg: Config,
    client: httpx.AsyncClient,
    url: str,
    path: Path,
    *,
    accept: str = "text/html",
    accept_404: bool = False,
) -> FetchResult:
    if not cfg.force and path.exists() and path.stat().st_size > 0:
        return FetchResult(ok=True, status_code=200, url=url, path=path)

    try:
        r = await request_with_retries(
            client, "GET", url, headers={"Accept": accept}, accept_404=accept_404
        )
        if accept_404 and r.status_code == 404:
            await write_text(path, "")  # marker
            return FetchResult(
                ok=False, status_code=404, url=url, path=path, error="404"
            )
        await write_text(path, r.text)
        return FetchResult(ok=True, status_code=r.status_code, url=url, path=path)
    except Exception as e:
        logger.exception("TEXT fetch failed: %s", url)
        return FetchResult(ok=False, status_code=0, url=url, path=path, error=str(e))


async def fetch_attachment(
    cfg: Config, client: httpx.AsyncClient, url: str, path: Path
) -> FetchResult:
    if not cfg.force and path.exists() and path.stat().st_size > 0:
        return FetchResult(ok=True, status_code=200, url=url, path=path)

    try:
        async with client.stream(
            "GET", url, headers={"Accept": "*/*"}, timeout=60.0
        ) as r:
            if r.status_code == 404:
                await write_bytes(path, b"")
                return FetchResult(
                    ok=False, status_code=404, url=url, path=path, error="404"
                )
            if r.status_code in (429, 500, 502, 503, 504):
                # let request_with_retries handle these; stream does not use it, so do a small manual retry
                raise httpx.HTTPStatusError("transient", request=r.request, response=r)
            r.raise_for_status()

            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)

        return FetchResult(ok=True, status_code=200, url=url, path=path)
    except Exception as e:
        logger.exception("Attachment fetch failed: %s", url)
        return FetchResult(ok=False, status_code=0, url=url, path=path, error=str(e))


# -----------------------------
# Domain logic
# -----------------------------
def extract_dates(obj: Any) -> list[str]:
    found: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", x):
            found.add(x)
        elif isinstance(x, list):
            for it in x:
                walk(it)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)

    if isinstance(obj, dict):
        for key in ("dates", "days", "sittingDays", "sessionDays"):
            if key in obj:
                walk(obj[key])
    walk(obj)
    return sorted(found)


async def download_mps_and_clubs(
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    mp_list_url = f"{cfg.api_root}/sejm/term{cfg.term}/MP"
    clubs_url = f"{cfg.api_root}/sejm/term{cfg.term}/clubs"
    mp_list_path = cfg.raw_dir / "mp" / "mp_list.json"
    clubs_path = cfg.raw_dir / "clubs" / "clubs.json"

    async with make_client(cfg) as client:
        await fetch_json_to_file(cfg, client, mp_list_url, mp_list_path)
        await fetch_json_to_file(cfg, client, clubs_url, clubs_path)

    mp_list = json.loads(mp_list_path.read_text(encoding="utf-8"))
    clubs_list = json.loads(clubs_path.read_text(encoding="utf-8"))

    # MP details jobs
    jobs: list[tuple[str, Path]] = []
    for mp in mp_list:
        mp_id = safe_int(mp.get("id") or mp.get("ID") or mp.get("mpId"))
        if not mp_id:
            continue
        url = f"{cfg.api_root}/sejm/term{cfg.term}/MP/{mp_id}"
        path = cfg.raw_dir / "mp" / "details" / f"{mp_id}.json"
        jobs.append((url, path))

    async def handler(client: httpx.AsyncClient, job: tuple[str, Path]) -> FetchResult:
        url, path = job
        return await fetch_json_to_file(cfg, client, url, path, accept_404=True)

    logger.info("MP details to fetch: %d", len(jobs))
    ok, fail = await run_job_pool(cfg, jobs, handler, desc="MP details (JSON)")
    logger.info("MP details OK=%d FAIL=%d", ok, fail)

    mp_details: dict[int, dict[str, Any]] = {}
    for p in (cfg.raw_dir / "mp" / "details").glob("*.json"):
        try:
            mp_id = safe_int(p.stem)
            if mp_id is None:
                continue
            if p.stat().st_size == 0:
                continue
            mp_details[mp_id] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

    return mp_list, mp_details, clubs_list


def build_dim_tables(
    cfg: Config,
    mp_list: list[dict[str, Any]],
    mp_details: dict[int, dict[str, Any]],
    clubs_list: list[dict[str, Any]],
) -> tuple[dict[int, str], dict[int, str]]:
    mp_dim_rows: list[dict[str, Any]] = []

    for mp in mp_list:
        mp_id = safe_int(mp.get("id") or mp.get("ID") or mp.get("mpId"))
        if not mp_id:
            continue
        det = mp_details.get(mp_id, {})
        mp_dim_rows.append(
            {
                "mp_id": mp_id,
                "name": mp.get("firstLastName")
                or mp.get("name")
                or det.get("firstLastName")
                or "",
                "club": mp.get("club") or det.get("club") or "",
                "districtName": det.get("districtName") or "",
                "voivodeship": det.get("voivodeship") or "",
                "birthDate": det.get("birthDate") or "",
                "birthPlace": det.get("birthPlace") or "",
                "educationLevel": det.get("educationLevel")
                or det.get("education")
                or "",
                "profession": det.get("profession") or "",
            }
        )

    if mp_dim_rows:
        mp_dim_path = cfg.tables_dir / "dim_mp.csv"
        write_csv(mp_dim_path, mp_dim_rows, list(mp_dim_rows[0].keys()))
        logger.info("Wrote: %s", mp_dim_path)

    club_rows: list[dict[str, Any]] = []
    for c in clubs_list:
        club_rows.append(
            {
                "club_id": c.get("id") or "",
                "name": c.get("name") or "",
                "membersCount": c.get("membersCount") or "",
            }
        )
    club_dim_path = cfg.tables_dir / "dim_club.csv"
    write_csv(club_dim_path, club_rows, ["club_id", "name", "membersCount"])
    logger.info("Wrote: %s", club_dim_path)

    mp_id_to_name = {r["mp_id"]: r["name"] for r in mp_dim_rows}
    mp_id_to_slug = {mp_id: slugify(name) for mp_id, name in mp_id_to_name.items()}
    return mp_id_to_name, mp_id_to_slug


async def download_proceedings_and_transcript_indexes(
    cfg: Config,
) -> list[tuple[int, str]]:
    proceedings_url = f"{cfg.api_root}/sejm/term{cfg.term}/proceedings"
    proceedings_path = cfg.raw_dir / "proceedings" / "proceedings.json"

    async with make_client(cfg) as client:
        await fetch_json_to_file(cfg, client, proceedings_url, proceedings_path)

    proceedings = json.loads(proceedings_path.read_text(encoding="utf-8"))

    # unique proceeding numbers
    nums: list[int] = []
    for p in proceedings:
        n = safe_int(
            p.get("num")
            or p.get("number")
            or p.get("proceedingNum")
            or p.get("sitting")
            or p.get("id")
        )
        if n is not None:
            nums.append(n)
    nums = sorted(set(nums))
    logger.info("Proceedings discovered: %d", len(nums))

    # fetch proceeding details (best-effort)
    detail_jobs: list[tuple[str, Path]] = []
    for n in nums:
        url = f"{cfg.api_root}/sejm/term{cfg.term}/proceedings/{n}"
        path = cfg.raw_dir / "proceedings" / "details" / f"{n}.json"
        detail_jobs.append((url, path))

    async def detail_handler(
        client: httpx.AsyncClient, job: tuple[str, Path]
    ) -> FetchResult:
        url, path = job
        return await fetch_json_to_file(cfg, client, url, path, accept_404=True)

    ok, fail = await run_job_pool(
        cfg, detail_jobs, detail_handler, desc="Proceeding details (JSON)"
    )
    logger.info("Proceeding details OK=%d FAIL=%d", ok, fail)

    # build proceeding-day pairs
    day_pairs: set[tuple[int, str]] = set()

    for n in nums:
        det_path = cfg.raw_dir / "proceedings" / "details" / f"{n}.json"
        dates: list[str] = []

        if det_path.exists() and det_path.stat().st_size > 0:
            try:
                det = json.loads(det_path.read_text(encoding="utf-8"))
                dates = extract_dates(det)
            except Exception:
                dates = []

        if not dates:
            # fallback: try to find dates in proceedings list entry
            for p in proceedings:
                pn = safe_int(
                    p.get("num")
                    or p.get("number")
                    or p.get("proceedingNum")
                    or p.get("sitting")
                    or p.get("id")
                )
                if pn == n:
                    dates = extract_dates(p)
                    break

        for d in dates:
            day_pairs.add((n, d))

    pairs = sorted(day_pairs)
    logger.info("Proceeding-day pairs: %d", len(pairs))

    # fetch transcript index per day
    idx_jobs: list[tuple[str, Path]] = []
    for n, d in pairs:
        url = f"{cfg.api_root}/sejm/term{cfg.term}/proceedings/{n}/{d}/transcripts"
        path = cfg.raw_dir / "transcripts" / "index_by_day" / f"p{n}" / f"d{d}.json"
        idx_jobs.append((url, path))

    async def idx_handler(
        client: httpx.AsyncClient, job: tuple[str, Path]
    ) -> FetchResult:
        url, path = job
        return await fetch_json_to_file(cfg, client, url, path, accept_404=True)

    ok, fail = await run_job_pool(
        cfg, idx_jobs, idx_handler, desc="Transcript indexes (JSON)"
    )
    logger.info("Transcript indexes OK=%d FAIL=%d", ok, fail)

    return pairs


def build_transcript_statement_index(
    cfg: Config, mp_id_to_slug: dict[int, str]
) -> tuple[list[dict[str, Any]], list[tuple[str, Path]]]:
    base = cfg.raw_dir / "transcripts" / "index_by_day"
    idx_files = list(base.rglob("d*.json"))

    statement_rows: list[dict[str, Any]] = []
    html_jobs: list[tuple[str, Path]] = []

    for idx_path in idx_files:
        try:
            if idx_path.stat().st_size == 0:
                continue
            obj = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        proceeding_num = safe_int(obj.get("proceedingNum"))
        date = obj.get("date")
        statements = obj.get("statements") or []

        if not proceeding_num or not date:
            continue

        for st in statements:
            st_num = st.get("num")
            mp_id = safe_int(st.get("memberID"))
            name = st.get("name") or ""
            function = st.get("function") or ""
            unspoken = st.get("unspoken")

            statement_rows.append(
                {
                    "term": cfg.term,
                    "proceedingNum": proceeding_num,
                    "date": date,
                    "statementNum": st_num,
                    "memberID": mp_id if mp_id is not None else "",
                    "name": name,
                    "function": function,
                    "startDateTime": st.get("startDateTime") or "",
                    "endDateTime": st.get("endDateTime") or "",
                    "rapporteur": st.get("rapporteur")
                    if st.get("rapporteur") is not None
                    else "",
                    "secretary": st.get("secretary")
                    if st.get("secretary") is not None
                    else "",
                    "unspoken": unspoken if unspoken is not None else "",
                }
            )

            st_num_str = str(st_num)
            url = f"{cfg.api_root}/sejm/term{cfg.term}/proceedings/{proceeding_num}/{date}/transcripts/{st_num_str}"

            if mp_id and mp_id > 0:
                mp_folder = f"{mp_id:03d}-{mp_id_to_slug.get(mp_id, slugify(name))}"
                path = (
                    cfg.raw_dir
                    / "transcripts"
                    / "html_by_mp"
                    / mp_folder
                    / f"p{proceeding_num}"
                    / f"d{date}"
                    / f"s{st_num_str}.html"
                )
            else:
                path = (
                    cfg.raw_dir
                    / "transcripts"
                    / "html_non_mp"
                    / f"p{proceeding_num}"
                    / f"d{date}"
                    / f"s{st_num_str}.html"
                )

            html_jobs.append((url, path))

    # Write CSV index
    fact_path = cfg.tables_dir / "fact_transcript_statement_index.csv"
    if statement_rows:
        write_csv(
            fact_path,
            statement_rows,
            [
                "term",
                "proceedingNum",
                "date",
                "statementNum",
                "memberID",
                "name",
                "function",
                "startDateTime",
                "endDateTime",
                "rapporteur",
                "secretary",
                "unspoken",
            ],
        )
        logger.info("Wrote: %s (rows=%d)", fact_path, len(statement_rows))

    return statement_rows, html_jobs


async def download_many_html(
    cfg: Config, jobs: list[tuple[str, Path]], desc: str
) -> tuple[int, int]:
    async def handler(client: httpx.AsyncClient, job: tuple[str, Path]) -> FetchResult:
        url, path = job
        return await fetch_text_to_file(
            cfg, client, url, path, accept="text/html", accept_404=True
        )

    return await run_job_pool(cfg, jobs, handler, desc=desc)


async def paginated_list(
    cfg: Config, client: httpx.AsyncClient, url: str, page_dir: Path
) -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    offset = 0
    limit = cfg.page_limit

    while True:
        page_path = page_dir / f"offset={offset}_limit={limit}.json"

        if not cfg.force and page_path.exists() and page_path.stat().st_size > 0:
            items = json.loads(page_path.read_text(encoding="utf-8"))
        else:
            r = await request_with_retries(
                client,
                "GET",
                url,
                headers={"Accept": "application/json"},
                params={"offset": offset, "limit": limit},
            )
            items = r.json()
            await write_json(page_path, items)

        if not items:
            break

        all_items.extend(items)
        if isinstance(items, list) and len(items) < limit:
            break

        offset += limit

    return all_items


async def download_interpellations(cfg: Config, mp_id_to_slug: dict[int, str]) -> None:
    base_list_url = f"{cfg.api_root}/sejm/term{cfg.term}/interpellations"
    page_dir = cfg.raw_dir / "interpellations" / "pages"

    async with make_client(cfg) as client:
        items = await paginated_list(cfg, client, base_list_url, page_dir)

    logger.info("Interpellations list items: %d", len(items))
    nums = sorted(
        {safe_int(it.get("num")) for it in items if safe_int(it.get("num")) is not None}
    )
    logger.info("Interpellations unique nums: %d", len(nums))

    # details jobs
    detail_jobs: list[tuple[str, Path]] = []
    for n in nums:
        url = f"{cfg.api_root}/sejm/term{cfg.term}/interpellations/{n}"
        path = cfg.raw_dir / "interpellations" / "by_num" / f"{n}" / "details.json"
        detail_jobs.append((url, path))

    async def details_handler(
        client: httpx.AsyncClient, job: tuple[str, Path]
    ) -> FetchResult:
        url, path = job
        return await fetch_json_to_file(cfg, client, url, path, accept_404=True)

    ok, fail = await run_job_pool(
        cfg, detail_jobs, details_handler, desc="Interpellations details (JSON)"
    )
    logger.info("Interpellations details OK=%d FAIL=%d", ok, fail)

    # build body/reply/attachment jobs + author pointers
    body_jobs: list[tuple[str, Path]] = []
    reply_jobs: list[tuple[str, Path]] = []
    attach_jobs: list[tuple[str, Path]] = []
    author_ptr_rows: list[dict[str, Any]] = []

    for n in nums:
        det_path = cfg.raw_dir / "interpellations" / "by_num" / f"{n}" / "details.json"
        if not det_path.exists() or det_path.stat().st_size == 0:
            continue
        try:
            det = json.loads(det_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        authors = det.get("from") or []
        author_ids: list[int] = []
        for a in authors:
            ai = safe_int(a)
            if ai is not None:
                author_ids.append(ai)

        for ai in author_ids:
            ptr_dir = (
                cfg.raw_dir
                / "interpellations"
                / "by_author_ptr"
                / f"{ai:03d}-{mp_id_to_slug.get(ai, 'unknown')}"
            )
            ptr_path = ptr_dir / f"i{n}.json"
            author_ptr_rows.append(
                {
                    "mp_id": ai,
                    "num": n,
                    "ptr_path": str(ptr_path.relative_to(cfg.raw_dir)),
                }
            )
            if cfg.force or not ptr_path.exists():
                await write_json(
                    ptr_path,
                    {
                        "type": "interpellation",
                        "num": n,
                        "target": str(det_path.parent),
                    },
                )

        body_url = f"{cfg.api_root}/sejm/term{cfg.term}/interpellations/{n}/body"
        body_path = cfg.raw_dir / "interpellations" / "by_num" / f"{n}" / "body.html"
        body_jobs.append((body_url, body_path))

        for rep in det.get("replies") or []:
            key = rep.get("key")
            if not key:
                continue
            rep_url = f"{cfg.api_root}/sejm/term{cfg.term}/interpellations/{n}/reply/{key}/body"
            rep_path = (
                cfg.raw_dir
                / "interpellations"
                / "by_num"
                / f"{n}"
                / "replies"
                / f"{key}.html"
            )
            reply_jobs.append((rep_url, rep_path))

            if cfg.download_attachments:
                for att in rep.get("attachments") or []:
                    aurl = att.get("URL")
                    aname = att.get("name") or "attachment.bin"
                    if aurl:
                        apath = (
                            cfg.raw_dir
                            / "interpellations"
                            / "by_num"
                            / f"{n}"
                            / "attachments"
                            / f"{key}"
                            / aname
                        )
                        attach_jobs.append((aurl, apath))

    if author_ptr_rows:
        ptr_index_path = cfg.tables_dir / "ptr_interpellations_by_author.csv"
        write_csv(ptr_index_path, author_ptr_rows, ["mp_id", "num", "ptr_path"])
        logger.info("Wrote: %s", ptr_index_path)

    ok1, fail1 = await download_many_html(
        cfg, body_jobs, "Interpellations bodies (HTML)"
    )
    ok2, fail2 = await download_many_html(
        cfg, reply_jobs, "Interpellations replies (HTML)"
    )
    logger.info(
        "Interpellations body OK=%d FAIL=%d; reply OK=%d FAIL=%d",
        ok1,
        fail1,
        ok2,
        fail2,
    )

    if cfg.download_attachments and attach_jobs:

        async def attach_handler(
            client: httpx.AsyncClient, job: tuple[str, Path]
        ) -> FetchResult:
            url, path = job
            return await fetch_attachment(cfg, client, url, path)

        ok_a, fail_a = await run_job_pool(
            cfg, attach_jobs, attach_handler, desc="Interpellations attachments"
        )
        logger.info("Interpellations attachments OK=%d FAIL=%d", ok_a, fail_a)


async def download_written_questions(
    cfg: Config, mp_id_to_slug: dict[int, str]
) -> None:
    base_list_url = f"{cfg.api_root}/sejm/term{cfg.term}/writtenQuestions"
    page_dir = cfg.raw_dir / "writtenQuestions" / "pages"

    async with make_client(cfg) as client:
        items = await paginated_list(cfg, client, base_list_url, page_dir)

    logger.info("writtenQuestions list items: %d", len(items))
    nums = sorted(
        {safe_int(it.get("num")) for it in items if safe_int(it.get("num")) is not None}
    )
    logger.info("writtenQuestions unique nums: %d", len(nums))

    detail_jobs: list[tuple[str, Path]] = []
    for n in nums:
        url = f"{cfg.api_root}/sejm/term{cfg.term}/writtenQuestions/{n}"
        path = cfg.raw_dir / "writtenQuestions" / "by_num" / f"{n}" / "details.json"
        detail_jobs.append((url, path))

    async def details_handler(
        client: httpx.AsyncClient, job: tuple[str, Path]
    ) -> FetchResult:
        url, path = job
        return await fetch_json_to_file(cfg, client, url, path, accept_404=True)

    ok, fail = await run_job_pool(
        cfg, detail_jobs, details_handler, desc="writtenQuestions details (JSON)"
    )
    logger.info("writtenQuestions details OK=%d FAIL=%d", ok, fail)

    body_jobs: list[tuple[str, Path]] = []
    reply_jobs: list[tuple[str, Path]] = []
    attach_jobs: list[tuple[str, Path]] = []
    author_ptr_rows: list[dict[str, Any]] = []

    for n in nums:
        det_path = cfg.raw_dir / "writtenQuestions" / "by_num" / f"{n}" / "details.json"
        if not det_path.exists() or det_path.stat().st_size == 0:
            continue
        try:
            det = json.loads(det_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        authors = det.get("from") or []
        author_ids: list[int] = []
        for a in authors:
            ai = safe_int(a)
            if ai is not None:
                author_ids.append(ai)

        for ai in author_ids:
            ptr_dir = (
                cfg.raw_dir
                / "writtenQuestions"
                / "by_author_ptr"
                / f"{ai:03d}-{mp_id_to_slug.get(ai, 'unknown')}"
            )
            ptr_path = ptr_dir / f"q{n}.json"
            author_ptr_rows.append(
                {
                    "mp_id": ai,
                    "num": n,
                    "ptr_path": str(ptr_path.relative_to(cfg.raw_dir)),
                }
            )
            if cfg.force or not ptr_path.exists():
                await write_json(
                    ptr_path,
                    {
                        "type": "writtenQuestion",
                        "num": n,
                        "target": str(det_path.parent),
                    },
                )

        body_url = f"{cfg.api_root}/sejm/term{cfg.term}/writtenQuestions/{n}/body"
        body_path = cfg.raw_dir / "writtenQuestions" / "by_num" / f"{n}" / "body.html"
        body_jobs.append((body_url, body_path))

        for rep in det.get("replies") or []:
            key = rep.get("key")
            if not key:
                continue
            rep_url = f"{cfg.api_root}/sejm/term{cfg.term}/writtenQuestions/{n}/reply/{key}/body"
            rep_path = (
                cfg.raw_dir
                / "writtenQuestions"
                / "by_num"
                / f"{n}"
                / "replies"
                / f"{key}.html"
            )
            reply_jobs.append((rep_url, rep_path))

            if cfg.download_attachments:
                for att in rep.get("attachments") or []:
                    aurl = att.get("URL")
                    aname = att.get("name") or "attachment.bin"
                    if aurl:
                        apath = (
                            cfg.raw_dir
                            / "writtenQuestions"
                            / "by_num"
                            / f"{n}"
                            / "attachments"
                            / f"{key}"
                            / aname
                        )
                        attach_jobs.append((aurl, apath))

    if author_ptr_rows:
        ptr_index_path = cfg.tables_dir / "ptr_writtenQuestions_by_author.csv"
        write_csv(ptr_index_path, author_ptr_rows, ["mp_id", "num", "ptr_path"])
        logger.info("Wrote: %s", ptr_index_path)

    ok1, fail1 = await download_many_html(
        cfg, body_jobs, "writtenQuestions bodies (HTML)"
    )
    ok2, fail2 = await download_many_html(
        cfg, reply_jobs, "writtenQuestions replies (HTML)"
    )
    logger.info(
        "writtenQuestions body OK=%d FAIL=%d; reply OK=%d FAIL=%d",
        ok1,
        fail1,
        ok2,
        fail2,
    )

    if cfg.download_attachments and attach_jobs:

        async def attach_handler(
            client: httpx.AsyncClient, job: tuple[str, Path]
        ) -> FetchResult:
            url, path = job
            return await fetch_attachment(cfg, client, url, path)

        ok_a, fail_a = await run_job_pool(
            cfg, attach_jobs, attach_handler, desc="writtenQuestions attachments"
        )
        logger.info("writtenQuestions attachments OK=%d FAIL=%d", ok_a, fail_a)


async def download_committees(cfg: Config) -> None:
    committees_url = f"{cfg.api_root}/sejm/term{cfg.term}/committees"
    committees_path = cfg.raw_dir / "committees" / "committees.json"

    async with make_client(cfg) as client:
        await fetch_json_to_file(
            cfg, client, committees_url, committees_path, accept_404=True
        )

    if not committees_path.exists() or committees_path.stat().st_size == 0:
        logger.warning("No committees list; skipping.")
        return

    committees = json.loads(committees_path.read_text(encoding="utf-8"))

    # fetch sittings list per committee
    sitting_jobs: list[tuple[str, Path]] = []
    for c in committees:
        code = c.get("code") or c.get("id") or c.get("committeeCode")
        if not code:
            continue
        url = f"{cfg.api_root}/sejm/term{cfg.term}/committees/{code}/sittings"
        path = cfg.raw_dir / "committees" / "sittings" / f"{code}_sittings.json"
        sitting_jobs.append((url, path))

    async def sitlist_handler(
        client: httpx.AsyncClient, job: tuple[str, Path]
    ) -> FetchResult:
        url, path = job
        return await fetch_json_to_file(cfg, client, url, path, accept_404=True)

    ok, fail = await run_job_pool(
        cfg, sitting_jobs, sitlist_handler, desc="Committees sittings lists (JSON)"
    )
    logger.info("Committees sittings list OK=%d FAIL=%d", ok, fail)

    # enumerate all sittings
    sitting_pairs: set[tuple[str, int]] = set()
    for p in (cfg.raw_dir / "committees" / "sittings").glob("*_sittings.json"):
        if p.stat().st_size == 0:
            continue
        code = p.name.replace("_sittings.json", "")
        try:
            sittings = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for s in sittings or []:
            sn = safe_int(s.get("num") or s.get("number") or s.get("id"))
            if sn is not None:
                sitting_pairs.add((code, sn))

    pairs = sorted(sitting_pairs)
    logger.info("Committee sittings discovered: %d", len(pairs))

    # build HTML/PDF jobs
    html_jobs: list[tuple[str, Path]] = []
    pdf_jobs: list[tuple[str, Path]] = []
    for code, sn in pairs:
        html_url = (
            f"{cfg.api_root}/sejm/term{cfg.term}/committees/{code}/sittings/{sn}/html"
        )
        pdf_url = (
            f"{cfg.api_root}/sejm/term{cfg.term}/committees/{code}/sittings/{sn}/pdf"
        )
        html_path = (
            cfg.raw_dir / "committees" / "sittings" / code / f"{sn}" / "sitting.html"
        )
        pdf_path = (
            cfg.raw_dir / "committees" / "sittings" / code / f"{sn}" / "sitting.pdf"
        )
        html_jobs.append((html_url, html_path))
        pdf_jobs.append((pdf_url, pdf_path))

    ok_h, fail_h = await download_many_html(cfg, html_jobs, "Committee sittings (HTML)")
    logger.info("Committee sittings HTML OK=%d FAIL=%d", ok_h, fail_h)

    async def pdf_handler(
        client: httpx.AsyncClient, job: tuple[str, Path]
    ) -> FetchResult:
        url, path = job
        return await fetch_attachment(cfg, client, url, path)

    ok_p, fail_p = await run_job_pool(
        cfg, pdf_jobs, pdf_handler, "Committee sittings (PDF)"
    )
    logger.info("Committee sittings PDF OK=%d FAIL=%d", ok_p, fail_p)


def count_files(root: Path, pattern: str) -> int:
    return sum(1 for _ in root.rglob(pattern))


# -----------------------------
# Main pipeline
# -----------------------------
async def main_async(cfg: Config) -> int:
    ensure_dirs(cfg)
    logger.info("Project root: %s", cfg.project_root)
    logger.info("Raw dir: %s", cfg.raw_dir)
    logger.info("Tables dir: %s", cfg.tables_dir)

    mp_list, mp_details, clubs_list = await download_mps_and_clubs(cfg)
    logger.info(
        "MP list=%d, MP details=%d, clubs=%d",
        len(mp_list),
        len(mp_details),
        len(clubs_list),
    )

    _, mp_id_to_slug = build_dim_tables(cfg, mp_list, mp_details, clubs_list)

    # transcripts
    if cfg.download_transcripts:
        await download_proceedings_and_transcript_indexes(cfg)
        _, html_jobs = build_transcript_statement_index(cfg, mp_id_to_slug)
        logger.info("Transcript HTML jobs: %d", len(html_jobs))
        ok, fail = await download_many_html(cfg, html_jobs, "Transcripts (HTML)")
        logger.info("Transcripts HTML OK=%d FAIL=%d", ok, fail)
    else:
        logger.info("Skipping transcripts.")

    # interpellations
    if cfg.download_interpellations:
        await download_interpellations(cfg, mp_id_to_slug)
    else:
        logger.info("Skipping interpellations.")

    # written questions
    if cfg.download_written_questions:
        await download_written_questions(cfg, mp_id_to_slug)
    else:
        logger.info("Skipping writtenQuestions.")

    # committees (optional heavy)
    if cfg.download_committees:
        await download_committees(cfg)
    else:
        logger.info("Skipping committees.")

    # quick validation summary
    logger.info("Validation summary:")
    logger.info(
        "  MP details: %d", count_files(cfg.raw_dir / "mp" / "details", "*.json")
    )
    logger.info(
        "  Transcript indexes: %d",
        count_files(cfg.raw_dir / "transcripts" / "index_by_day", "*.json"),
    )
    logger.info(
        "  Transcript HTML (MP): %d",
        count_files(cfg.raw_dir / "transcripts" / "html_by_mp", "*.html"),
    )
    logger.info(
        "  Interpellations bodies: %d",
        count_files(cfg.raw_dir / "interpellations" / "by_num", "body.html"),
    )
    logger.info(
        "  writtenQuestions bodies: %d",
        count_files(cfg.raw_dir / "writtenQuestions" / "by_num", "body.html"),
    )

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Sejm texts (term-based) to local corpus."
    )
    p.add_argument("--term", type=int, default=10, help="Sejm term, e.g. 10")
    p.add_argument("--api-root", default="https://api.sejm.gov.pl", help="API root")
    p.add_argument(
        "--project-root", default=".", help="Project root (default: current dir)"
    )
    p.add_argument(
        "--user-agent",
        default="sejm-textmining/0.1 (research)",
        help="User-Agent header",
    )

    p.add_argument(
        "--concurrency", type=int, default=60, help="Async worker count (default 60)"
    )
    p.add_argument(
        "--max-connections",
        type=int,
        default=200,
        help="HTTP connection pool max (default 200)",
    )
    p.add_argument(
        "--max-keepalive", type=int, default=50, help="Keepalive pool max (default 50)"
    )
    p.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help="Pagination limit for lists (default 100)",
    )

    p.add_argument("--http2", action="store_true", help="Enable HTTP/2 (default on)")
    p.add_argument(
        "--no-http2", dest="http2", action="store_false", help="Disable HTTP/2"
    )
    p.set_defaults(http2=True)

    p.add_argument(
        "--force", action="store_true", help="Redownload and overwrite cached files"
    )

    p.add_argument(
        "--no-transcripts",
        dest="download_transcripts",
        action="store_false",
        help="Skip transcripts",
    )
    p.add_argument(
        "--no-interpellations",
        dest="download_interpellations",
        action="store_false",
        help="Skip interpellations",
    )
    p.add_argument(
        "--no-written-questions",
        dest="download_written_questions",
        action="store_false",
        help="Skip writtenQuestions",
    )
    p.add_argument(
        "--no-attachments",
        dest="download_attachments",
        action="store_false",
        help="Skip attachments",
    )
    p.add_argument(
        "--committees",
        dest="download_committees",
        action="store_true",
        help="Also download committee sittings (heavy)",
    )

    p.set_defaults(
        download_transcripts=True,
        download_interpellations=True,
        download_written_questions=True,
        download_attachments=True,
        download_committees=False,
    )
    return p.parse_args()


def setup_logging(logs_dir: Path, term: int) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"sejm_term{term}_{run_tag}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("Log file: %s", log_path)
    return log_path


def build_config(args: argparse.Namespace) -> Config:
    project_root = Path(args.project_root).expanduser().resolve()
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw" / f"term{args.term}"
    tables_dir = data_dir / "tables" / f"term{args.term}"
    logs_dir = project_root / "logs"

    return Config(
        term=args.term,
        api_root=args.api_root,
        user_agent=args.user_agent,
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        tables_dir=tables_dir,
        logs_dir=logs_dir,
        http2=bool(args.http2),
        concurrency=int(args.concurrency),
        max_connections=int(args.max_connections),
        max_keepalive=int(args.max_keepalive),
        page_limit=int(args.page_limit),
        force=bool(args.force),
        download_transcripts=bool(args.download_transcripts),
        download_interpellations=bool(args.download_interpellations),
        download_written_questions=bool(args.download_written_questions),
        download_attachments=bool(args.download_attachments),
        download_committees=bool(args.download_committees),
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    setup_logging(cfg.logs_dir, cfg.term)

    ensure_dirs(cfg)

    try:
        return asyncio.run(main_async(cfg))
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
