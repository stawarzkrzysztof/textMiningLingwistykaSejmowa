import os

import requests
from tqdm import tqdm

out_dir = "/Users/krzysztofstawarz/GithubRepositories/textMiningLingwistykaSejmowa/1dataDownload/data/raw/term10/mp/images"

for mpID in tqdm(range(1, 499)):
    url = f"https://api.sejm.gov.pl/sejm/term10/MP/{mpID}/photo"

    out_path = os.path.join(out_dir, f"{mpID:03d}.jpg")
    if os.path.exists(out_path):
        continue

    resp = requests.get(url, headers={"Accept": "image/jpeg"}, timeout=30)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith(
        "image/"
    ):
        with open(out_path, "wb") as f:
            f.write(resp.content)
    else:
        print(f"[WARN] {mpID}: {resp.status_code}")
