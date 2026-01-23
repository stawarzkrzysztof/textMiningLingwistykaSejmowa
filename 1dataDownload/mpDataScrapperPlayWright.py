#!/usr/bin/env python3
"""
Human-behavior-ish scraper for Sejm 10th-term MP pages (posel.xsp?id=001&type=A ... 498).

Scrapes:
- name (top header),
- "Staż parlamentarny:",
- "Ukończona szkoła:"

Outputs: 1.json, 2.json, ... 498.json

Notes:
- sejm.gov.pl may show an anti-bot verification page. This script tries to minimize
  triggers and backs off automatically. If still blocked, it will ask you to solve it
  once in the opened browser window and then continue.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


BASE_URL = "https://www.sejm.gov.pl/Sejm10.nsf/posel.xsp?id={id3}&type=A"


def _collapse_ws(s: str) -> str:
    return " ".join(s.split()).strip()


def human_pause(min_s: float, max_s: float) -> None:
    """Randomized pause to mimic human pacing."""
    time.sleep(random.uniform(min_s, max_s))


def page_is_blocked_html(html: str) -> bool:
    markers = [
        "Please enable JavaScript",
        "What code is in the image",
        "testing whether you are a human visitor",
        "Audio is not supported in your browser",
        "Press & Hold",
        "captcha",
    ]
    low = html.lower()
    return any(m.lower() in low for m in markers)


def page_is_blocked(page) -> bool:
    try:
        return page_is_blocked_html(page.content())
    except Exception:
        return False


def light_human_interaction(page) -> None:
    """
    Very light interaction: move mouse a bit and scroll.
    Keep it subtle; too much "randomness" can also look botty.
    """
    try:
        # Small mouse move within viewport
        page.mouse.move(
            random.randint(50, 400),
            random.randint(80, 500),
            steps=random.randint(8, 20),
        )
    except Exception:
        pass

    # Scroll in a couple of steps
    try:
        for _ in range(random.randint(1, 3)):
            page.mouse.wheel(0, random.randint(300, 900))
            human_pause(0.15, 0.6)
        # Occasionally scroll back up slightly
        if random.random() < 0.25:
            page.mouse.wheel(0, -random.randint(200, 500))
            human_pause(0.1, 0.4)
    except Exception:
        pass


def extract_name(page) -> Optional[str]:
    # Try header-like elements; fallback to title if needed
    selectors = [
        "css=h1",
        "css=h2",
        "xpath=(//h1|//h2)[1]",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                t = _collapse_ws(loc.inner_text())
                if t and len(t) <= 80 and "Sejm" not in t:
                    return t
        except Exception:
            continue
    try:
        t = _collapse_ws(page.title())
        return t if t else None
    except Exception:
        return None


def extract_value_by_label(page, label: str) -> Optional[str]:
    xpaths = [
        f"xpath=//td[normalize-space()='{label}']/following-sibling::td[1]",
        f"xpath=//th[normalize-space()='{label}']/following-sibling::td[1]",
        f"xpath=//*[normalize-space()='{label}']/following-sibling::*[1]",
        f"xpath=//*[contains(normalize-space(), '{label}')]/ancestor::tr[1]/*[2]",
    ]
    for xp in xpaths:
        try:
            loc = page.locator(xp).first
            if loc.count() > 0:
                t = _collapse_ws(loc.inner_text())
                if t:
                    return t
        except Exception:
            continue
    return None


def goto_with_wait(page, url: str, timeout_ms: int) -> None:
    # Use domcontentloaded then a short settle time; networkidle can hang on some sites.
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    human_pause(0.3, 1.1)


def handle_block_with_backoff(page, url: str, max_backoff_rounds: int) -> None:
    """
    If blocked, wait (increasing backoff) and retry.
    Only if it stays blocked after backoff do we ask for manual solve.
    """
    backoff_seconds = 20.0
    for round_idx in range(1, max_backoff_rounds + 1):
        if not page_is_blocked(page):
            return

        # Backoff with jitter
        wait_s = backoff_seconds * random.uniform(0.8, 1.4)
        print(
            f"\nAnti-bot detected. Backing off for ~{int(wait_s)}s (round {round_idx}/{max_backoff_rounds})..."
        )
        time.sleep(wait_s)

        # Retry navigation
        try:
            goto_with_wait(page, url, timeout_ms=60_000)
        except Exception:
            pass

        # Increase backoff
        backoff_seconds *= 1.7

    # Still blocked -> manual solve once
    if page_is_blocked(page):
        print("\nStill seeing verification page after backoff.")
        print("Please solve the verification/captcha in the opened browser window.")
        input("Press ENTER here once completed...")
        goto_with_wait(page, url, timeout_ms=60_000)


def scrape_one(page, i: int) -> dict:
    url = BASE_URL.format(id3=str(i).zfill(3))

    goto_with_wait(page, url, timeout_ms=60_000)
    handle_block_with_backoff(page, url, max_backoff_rounds=3)

    # Light “reading” behavior
    light_human_interaction(page)

    name = extract_name(page)
    staz = extract_value_by_label(page, "Staż parlamentarny:")
    ukonczona = extract_value_by_label(page, "Ukończona szkoła:")

    return {
        "mp_num": i,
        "url": url,
        "name": name,
        "staz_parlamentarny": staz,
        "ukonczona_szkola": ukonczona,
    }


def maybe_take_long_break(counter: int, next_break_at: int) -> Tuple[bool, int]:
    """
    Occasionally take a longer break after N pages.
    Returns (took_break, new_next_break_at).
    """
    if counter < next_break_at:
        return (False, next_break_at)

    # Take a longer break: 15–60 seconds, rarely 2–4 minutes.
    if random.random() < 0.12:
        long_s = random.uniform(120, 240)
    else:
        long_s = random.uniform(15, 60)

    print(f"\nTaking a longer break (~{int(long_s)}s) to reduce bot-likeness...")
    time.sleep(long_s)

    # Schedule the next break after another 18–45 pages
    new_next = counter + random.randint(18, 45)
    return (True, new_next)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default="sejm10_mps_scraped",
        help="Output directory for JSON files.",
    )
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=498)
    ap.add_argument(
        "--profile-dir",
        default="pw_profile_sejm10",
        help="Persistent Playwright profile dir.",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Headless mode (not recommended for first run).",
    )
    ap.add_argument(
        "--use-installed-chrome",
        action="store_true",
        help="Use your installed Google Chrome via Playwright channel='chrome' (optional).",
    )
    ap.add_argument(
        "--min-delay", type=float, default=0.9, help="Min delay between MPs (seconds)."
    )
    ap.add_argument(
        "--max-delay", type=float, default=2.4, help="Max delay between MPs (seconds)."
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = range(args.start, args.end + 1)
    it = tqdm(rng, desc="Scraping MPs") if tqdm else rng

    # Plan first long break randomly after 18–45 pages
    next_break_at = random.randint(18, 45)
    processed = 0

    with sync_playwright() as p:
        launch_kwargs = dict(
            user_data_dir=args.profile_dir,
            headless=args.headless,
            locale="pl-PL",
            timezone_id="Europe/Warsaw",
            viewport={"width": 1365, "height": 768},
        )
        if args.use_installed_chrome:
            # Requires Chrome installed; does not need to be default browser.
            launch_kwargs["channel"] = "chrome"

        context = p.chromium.launch_persistent_context(**launch_kwargs)

        # Mildly “normal” headers
        context.set_extra_http_headers(
            {
                "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.7,en;q=0.6",
            }
        )

        page = context.new_page()
        page.set_default_timeout(60_000)

        # Start from a “normal” landing first (optional but helps some WAFs)
        try:
            goto_with_wait(
                page,
                "https://www.sejm.gov.pl/Sejm10.nsf/poslowie.xsp",
                timeout_ms=60_000,
            )
            human_pause(0.8, 2.0)
        except Exception:
            pass

        for i in it:
            out_path = outdir / f"{i}.json"
            if out_path.exists():
                continue

            # Between MPs: variable “reading/thinking” delay
            human_pause(args.min_delay, args.max_delay)

            # Occasionally take a longer break
            processed += 1
            _, next_break_at = maybe_take_long_break(processed, next_break_at)

            # Retry with increasing wait if transient failures happen
            last_err = None
            for attempt in range(1, 4):
                try:
                    data = scrape_one(page, i)
                    out_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    last_err = None
                    break
                except PlaywrightTimeoutError as e:
                    last_err = e
                    time.sleep(2.0 * attempt + random.uniform(0.0, 1.5))
                except Exception as e:
                    last_err = e
                    time.sleep(2.0 * attempt + random.uniform(0.0, 2.5))

            if last_err is not None:
                err_data = {
                    "mp_num": i,
                    "url": BASE_URL.format(id3=str(i).zfill(3)),
                    "error": repr(last_err),
                }
                out_path.write_text(
                    json.dumps(err_data, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        context.close()

    print(f"Done. JSON files are in: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
