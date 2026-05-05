from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://sid.erda.dk/"
ROOT_LISTING_URL = (
    "https://sid.erda.dk/cgi-sid/ls.py?share_id=f6hdp1zTzh&current_dir=.&flags=f"
)
SUPPORTED_TASKS = {"variant_effects", "histone_modification"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected BEND task files from ERDA.")
    parser.add_argument(
        "--tasks",
        default="variant_effects,histone_modification",
        help="Comma-separated BEND task directories to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/bend",
        help="Destination directory for downloaded files.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for ERDA requests.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def _request_with_retry(
    session: requests.Session,
    url: str,
    *,
    retries: int,
    timeout: float,
    stream: bool = False,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True, stream=stream)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            wait_seconds = min(2 ** (attempt - 1), 8)
            print(f"[WARN] Request failed ({exc}). Retrying in {wait_seconds}s: {url}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def _get_soup(
    session: requests.Session,
    url: str,
    *,
    retries: int,
    timeout: float,
) -> BeautifulSoup:
    response = _request_with_retry(
        session,
        url,
        retries=retries,
        timeout=timeout,
    )
    return BeautifulSoup(response.content, "lxml")


def _iter_directory_links(soup: BeautifulSoup) -> Iterable[tuple[str, str]]:
    for tag in soup.find_all("a", {"class": ["leftpad directoryicon"]}):
        name = tag.get_text(strip=True)
        href = tag.get("href")
        if not name or not href:
            continue
        yield name, urljoin(BASE_URL, f"cgi-sid/{href}")


def _iter_file_links(soup: BeautifulSoup) -> Iterable[tuple[str, str]]:
    for tag in soup.find_all("a", {"title": "open"}):
        href = tag.get("href")
        if not href:
            continue
        url = urljoin(BASE_URL, href)
        query = parse_qs(urlparse(url).query)
        filename = query.get("filename", [tag.get_text(strip=True)])[0]
        current_dir = query.get("current_dir", ["."])[0]
        relative_path = Path(current_dir) / filename if current_dir not in {"", "."} else Path(filename)
        yield relative_path.as_posix(), url


def _download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    retries: int,
    timeout: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] {destination}")
        return

    response = _request_with_retry(
        session,
        url,
        retries=retries,
        timeout=timeout,
        stream=True,
    )
    with destination.open("wb") as fp:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                fp.write(chunk)
    print(f"[ok] {destination}")


def _download_tree(
    session: requests.Session,
    listing_url: str,
    destination_root: Path,
    *,
    retries: int,
    timeout: float,
    visited: set[str],
) -> None:
    if listing_url in visited:
        return
    visited.add(listing_url)

    soup = _get_soup(session, listing_url, retries=retries, timeout=timeout)
    for relative_path, file_url in _iter_file_links(soup):
        _download_file(
            session,
            file_url,
            destination_root / relative_path,
            retries=retries,
            timeout=timeout,
        )

    for _, child_url in _iter_directory_links(soup):
        _download_tree(
            session,
            child_url,
            destination_root,
            retries=retries,
            timeout=timeout,
            visited=visited,
        )


def _resolve_task_links(
    session: requests.Session,
    tasks: list[str],
    *,
    retries: int,
    timeout: float,
) -> dict[str, str]:
    soup = _get_soup(session, ROOT_LISTING_URL, retries=retries, timeout=timeout)
    root_links = dict(_iter_directory_links(soup))
    missing = [task for task in tasks if task not in root_links]
    if missing:
        raise SystemExit(
            f"Could not find BEND task directories on ERDA: {', '.join(missing)}"
        )
    return {task: root_links[task] for task in tasks}


def main() -> None:
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unsupported = [task for task in tasks if task not in SUPPORTED_TASKS]
    if unsupported:
        raise SystemExit(
            f"Unsupported tasks: {', '.join(unsupported)}. Choose from: {sorted(SUPPORTED_TASKS)}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        task_links = _resolve_task_links(
            session,
            tasks,
            retries=args.retries,
            timeout=args.timeout,
        )
        for task, listing_url in task_links.items():
            print(f"Downloading {task} from {listing_url}")
            _download_tree(
                session,
                listing_url,
                output_dir,
                retries=args.retries,
                timeout=args.timeout,
                visited=set(),
            )


if __name__ == "__main__":
    main()