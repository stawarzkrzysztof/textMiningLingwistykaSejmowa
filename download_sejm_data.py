#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from tqdm import tqdm


BASE_URL = "https://api.sejm.gov.pl"


class SejmDownloader:
    def __init__(
        self,
        term: int,
        data_dir: Path,
        concurrency: int = 12,
        page_size: int = 500,
        download_pdfs: bool = True,
    ) -> None:
        self.term = term
        self.data_dir = data_dir / f"term{term}"
        self.concurrency = concurrency
        self.page_size = page_size
        self.download_pdfs = download_pdfs
        self.failures: List[Tuple[str, Path, Exception]] = []

        self.proceedings_dir = self.data_dir / "proceedings"
        self.interpellations_dir = self.data_dir / "interpellations"
        self.written_questions_dir = self.data_dir / "written_questions"
        self.meta_dir = self.data_dir / "meta"

        for path in [
            self.proceedings_dir,
            self.interpellations_dir,
            self.written_questions_dir,
            self.meta_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        limits = httpx.Limits(
            max_connections=self.concurrency * 2,
            max_keepalive_connections=self.concurrency,
        )
        timeout = httpx.Timeout(60.0, connect=15.0, read=60.0, write=60.0)
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            http2=True,
            follow_redirects=True,
            limits=limits,
            timeout=timeout,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def _absolute_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return f"{BASE_URL}{url}"

    async def _get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        retries: int = 3,
    ) -> httpx.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await asyncio.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def _fetch_paginated(
        self,
        url: str,
        label: str,
        save_dir: Path,
    ) -> List[Dict[str, Any]]:
        save_dir.mkdir(parents=True, exist_ok=True)
        offset = 0
        results: List[Dict[str, Any]] = []
        pbar = tqdm(desc=f"{label} index", unit="item")
        while True:
            response = await self._get(
                url,
                params={"limit": self.page_size, "offset": offset},
            )
            chunk_path = save_dir / f"{offset:06d}.json"
            if not chunk_path.exists():
                chunk_path.write_bytes(response.content)
            data = response.json()
            if not data:
                break
            results.extend(data)
            pbar.update(len(data))
            if len(data) < self.page_size:
                break
            offset += self.page_size
        pbar.close()
        return results

    async def _run_tasks(
        self,
        tasks: Iterable[Tuple[str, Path]],
        desc: str,
    ) -> None:
        task_list = list(tasks)
        if not task_list:
            return
        sem = asyncio.Semaphore(self.concurrency)
        pbar = tqdm(total=len(task_list), desc=desc)

        async def worker(url: str, dest: Path) -> None:
            if dest.exists():
                pbar.update(1)
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            success = False
            exc_obj: Optional[Exception] = None
            for attempt in range(3):
                try:
                    async with sem:
                        response = await self.client.get(self._absolute_url(url))
                    response.raise_for_status()
                    dest.write_bytes(response.content)
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    exc_obj = exc
                    await asyncio.sleep(1.5 * (attempt + 1))
            if not success and exc_obj is not None:
                self.failures.append((url, dest, exc_obj))
                pbar.write(f"[failed] {url} -> {dest} ({exc_obj})")
            pbar.update(1)

        await asyncio.gather(*(worker(url, dest) for url, dest in task_list))
        pbar.close()

    async def download_mp_list(self) -> None:
        path = self.meta_dir / "mp_list.json"
        if path.exists():
            return
        response = await self._get(f"/sejm/term{self.term}/MP")
        path.write_bytes(response.content)

    async def download_proceedings(self) -> None:
        list_resp = await self._get(f"/sejm/term{self.term}/proceedings")
        (self.proceedings_dir / "list.json").write_bytes(list_resp.content)
        proceedings = list_resp.json()

        statement_tasks: List[Tuple[str, Path]] = []
        pdf_tasks: List[Tuple[str, Path]] = []

        for proc in proceedings:
            proc_number = proc["number"]
            proc_dir = self.proceedings_dir / f"{proc_number:03d}"
            proc_dir.mkdir(parents=True, exist_ok=True)
            meta_path = proc_dir / "meta.json"
            if not meta_path.exists():
                meta_path.write_text(json.dumps(proc, ensure_ascii=False, indent=2))

            for date in proc.get("dates", []):
                date_dir = proc_dir / date
                date_dir.mkdir(parents=True, exist_ok=True)
                transcripts_url = (
                    f"/sejm/term{self.term}/proceedings/{proc_number}/{date}/transcripts"
                )
                transcripts_path = date_dir / "statements.json"
                if transcripts_path.exists():
                    try:
                        transcripts_json = json.loads(transcripts_path.read_text())
                    except Exception:  # noqa: BLE001
                        transcripts_resp = await self._get(transcripts_url)
                        transcripts_path.write_bytes(transcripts_resp.content)
                        transcripts_json = transcripts_resp.json()
                else:
                    try:
                        transcripts_resp = await self._get(transcripts_url)
                    except Exception as exc:  # noqa: BLE001
                        self.failures.append((transcripts_url, date_dir, exc))
                        continue
                    transcripts_path.write_bytes(transcripts_resp.content)
                    transcripts_json = transcripts_resp.json()

                for stmt in transcripts_json.get("statements", []):
                    stmt_num = stmt["num"]
                    dest = date_dir / "statements" / f"{stmt_num:05d}.html"
                    statement_tasks.append(
                        (
                            f"/sejm/term{self.term}/proceedings/{proc_number}/{date}/transcripts/{stmt_num}",
                            dest,
                        )
                    )
                if self.download_pdfs:
                    pdf_tasks.append(
                        (
                            f"/sejm/term{self.term}/proceedings/{proc_number}/{date}/transcripts/pdf",
                            date_dir / "transcript.pdf",
                        )
                    )

        await self._run_tasks(statement_tasks, "proceedings statements")
        if self.download_pdfs:
            await self._run_tasks(pdf_tasks, "proceedings pdfs")

    async def download_interpellations(self) -> None:
        items = await self._fetch_paginated(
            f"/sejm/term{self.term}/interpellations",
            "interpellations",
            self.interpellations_dir / "index",
        )
        tasks: List[Tuple[str, Path]] = []
        for item in items:
            num = item["num"]
            tasks.append(
                (
                    f"/sejm/term{self.term}/interpellations/{num}",
                    self.interpellations_dir / "details" / f"{num}.json",
                )
            )
            tasks.append(
                (
                    f"/sejm/term{self.term}/interpellations/{num}/body",
                    self.interpellations_dir / "bodies" / f"{num}.html",
                )
            )
            for att in item.get("attachments", []):
                url = att.get("URL") or att.get("url")
                if url:
                    fname = att.get("name") or Path(url).name
                    tasks.append(
                        (
                            url,
                            self.interpellations_dir
                            / "attachments"
                            / f"{num}"
                            / fname,
                        )
                    )
            for reply in item.get("replies", []):
                key = reply["key"]
                tasks.append(
                    (
                        f"/sejm/term{self.term}/interpellations/{num}/reply/{key}/body",
                        self.interpellations_dir / "replies" / f"{num}" / f"{key}.html",
                    )
                )
                for att in reply.get("attachments", []):
                    url = att.get("URL") or att.get("url")
                    if not url:
                        continue
                    fname = att.get("name") or Path(url).name
                    tasks.append(
                        (
                            url,
                            self.interpellations_dir
                            / "attachments"
                            / f"{num}"
                            / fname,
                        )
                    )
        await self._run_tasks(tasks, "interpellations bodies/replies/attachments")

    async def download_written_questions(self) -> None:
        items = await self._fetch_paginated(
            f"/sejm/term{self.term}/writtenQuestions",
            "written questions",
            self.written_questions_dir / "index",
        )
        tasks: List[Tuple[str, Path]] = []
        for item in items:
            num = item["num"]
            tasks.append(
                (
                    f"/sejm/term{self.term}/writtenQuestions/{num}",
                    self.written_questions_dir / "details" / f"{num}.json",
                )
            )
            tasks.append(
                (
                    f"/sejm/term{self.term}/writtenQuestions/{num}/body",
                    self.written_questions_dir / "bodies" / f"{num}.html",
                )
            )
            for att in item.get("attachments", []):
                url = att.get("URL") or att.get("url")
                if url:
                    fname = att.get("name") or Path(url).name
                    tasks.append(
                        (
                            url,
                            self.written_questions_dir
                            / "attachments"
                            / f"{num}"
                            / fname,
                        )
                    )
            for reply in item.get("replies", []):
                key = reply["key"]
                tasks.append(
                    (
                        f"/sejm/term{self.term}/writtenQuestions/{num}/reply/{key}/body",
                        self.written_questions_dir
                        / "replies"
                        / f"{num}"
                        / f"{key}.html",
                    )
                )
                for att in reply.get("attachments", []):
                    url = att.get("URL") or att.get("url")
                    if not url:
                        continue
                    fname = att.get("name") or Path(url).name
                    tasks.append(
                        (
                            url,
                            self.written_questions_dir
                            / "attachments"
                            / f"{num}"
                            / fname,
                        )
                    )
        await self._run_tasks(tasks, "written questions bodies/replies/attachments")

    async def run(
        self,
        skip_proceedings: bool = False,
        skip_interpellations: bool = False,
        skip_written: bool = False,
    ) -> None:
        try:
            await self.download_mp_list()
            if not skip_proceedings:
                await self.download_proceedings()
            if not skip_interpellations:
                await self.download_interpellations()
            if not skip_written:
                await self.download_written_questions()
        finally:
            await self.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pobiera surowe dane z API Sejmu (10. kadencja): "
            "stenogramy wystąpień, interpelacje, zapytania pisemne."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("sejm_raw"),
        help="Katalog bazowy na surowe dane (domyślnie: sejm_raw).",
    )
    parser.add_argument(
        "--term",
        type=int,
        default=10,
        help="Numer kadencji Sejmu do pobrania (domyślnie: 10).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="Liczba równoległych połączeń HTTP (domyślnie: 12).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Limit rekordów na jedno żądanie listujące (domyślnie: 500).",
    )
    parser.add_argument(
        "--no-proceedings-pdf",
        action="store_true",
        help="Pomiń pobieranie plików PDF stenogramów.",
    )
    parser.add_argument(
        "--skip-proceedings",
        action="store_true",
        help="Pomiń pobieranie stenogramów posiedzeń.",
    )
    parser.add_argument(
        "--skip-interpellations",
        action="store_true",
        help="Pomiń pobieranie interpelacji.",
    )
    parser.add_argument(
        "--skip-written",
        action="store_true",
        help="Pomiń pobieranie zapytań pisemnych.",
    )
    args = parser.parse_args()

    downloader = SejmDownloader(
        term=args.term,
        data_dir=args.data_dir,
        concurrency=args.concurrency,
        page_size=args.page_size,
        download_pdfs=not args.no_proceedings_pdf,
    )
    await downloader.run(
        skip_proceedings=args.skip_proceedings,
        skip_interpellations=args.skip_interpellations,
        skip_written=args.skip_written,
    )

    if downloader.failures:
        print("\nNiepowodzenia:")
        for url, dest, exc in downloader.failures:
            print(f"- {url} -> {dest}: {exc}")
    else:
        print("\nPobieranie zakończone bez błędów.")


if __name__ == "__main__":
    asyncio.run(main())
