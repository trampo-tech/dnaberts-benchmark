# adapted from https://github.com/frederikkemarin/BEND/blob/main/scripts/download_bend.py

from pathlib import Path

import hydra
import requests
from bs4 import BeautifulSoup
from omegaconf import DictConfig

BASE_LINK = "https://sid.erda.dk/"
SHARE_ID = "aNQa0Oz2lY"
SUPPORTED_TASKS = {"variant_effects", "histone_modification", "genomes"}
REDIRECT_PREFIX = f"/share_redirect/{SHARE_ID}/"


def get_soup(link):
    source_code = requests.get(link)
    soup = BeautifulSoup(source_code.content, "lxml")
    f = []
    f.extend(soup.find_all("a", {"class": ["leftpad directoryicon"]}))
    f.extend(soup.find_all("a", {"title": "open"}))
    return f


def download_file(link, destination):
    r = requests.get(link, allow_redirects=True)
    with open(destination, "wb") as fp:
        fp.write(r.content)


def relative_download_path(href):
    if href.startswith(REDIRECT_PREFIX):
        return href.removeprefix(REDIRECT_PREFIX)
    return href.lstrip("/")


def rec(link, destination: Path):
    f = get_soup(link)
    for child in f:
        if child.get("title") == "open":
            file_link = f'{BASE_LINK}{child.get("href")}'
            relative_path = relative_download_path(child.get("href"))
            if not relative_path:
                continue
            full_path = destination / relative_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            print(full_path)
            download_file(file_link, full_path)
        else:
            dir_link = f'{BASE_LINK}cgi-sid/{child.get("href")}'
            rec(dir_link, destination)


@hydra.main(version_base=None, config_path="../src/config", config_name="download_bend")
def main(cfg: DictConfig) -> None:
    tasks = [str(task).strip() for task in cfg.tasks if str(task).strip()]
    invalid_tasks = sorted(set(tasks) - SUPPORTED_TASKS)
    if invalid_tasks:
        raise SystemExit(
            f"Unsupported BEND tasks: {', '.join(invalid_tasks)}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_TASKS))}"
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        link = (
            f"{BASE_LINK}cgi-sid/ls.py?"
            f"share_id={SHARE_ID}&current_dir=data/{task}&flags=f"
        )
        print(f"DOWNLOADING {task} from {link}")
        rec(link, destination=output_dir)

    print("DONE")


if __name__ == "__main__":
    main()
