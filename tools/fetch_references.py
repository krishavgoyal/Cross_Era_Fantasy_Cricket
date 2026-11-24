"""Download freely licensed reference media from Wikimedia Commons into data/ and record the sources.

    python tools/fetch_references.py            # download what is missing, rewrite data/SOURCES_commons.md

Every file here is public domain or CC BY / CC BY-SA (attribution required; keep SOURCES_commons.md
with the files). Folders:
    data/reference_players/<player>/   photos of the player to animate (appearance)
    data/backgrounds/                  cricket grounds to put the player in
    data/motion_videos/<action>/       clips whose motion is transferred (skeleton only)
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = "CrossEraCricketStudentProject/1.0 (educational research; Wikimedia Commons API)"
API = "https://commons.wikimedia.org/w/api.php"

FILES = {
    # Don Bradman (batting reference)
    "Bradman&Bat.jpg": "reference_players/don_bradman/bradman_bat_sam_hood_pd.jpg",
    'Don Bradman with his "Don Bradman" brand Sykes bat, ca. 1932 - by Sam Hood (3064010387).jpg':
        "reference_players/don_bradman/bradman_sykes_bat_1932_sam_hood.jpg",
    "Don Bradman 1930.jpg": "reference_players/don_bradman/bradman_1930_slsa.jpg",
    # Backgrounds
    "SLNSW 8341 Test cricket at Sydney Cricket Ground.jpg": "backgrounds/scg_test_cricket_sam_hood_pd.jpg",
    "Sydney Cricket Ground 1930s.jpg": "backgrounds/scg_1930s_pd.jpg",
    "Lords-Cricket-Ground-Pavilion-06-08-2017.jpg": "backgrounds/lords_pavilion_2017.jpg",
    # Motion: bowling (net sessions, side-on, close)
    "Graham Onions bowling, 2012.ogv": "motion_videos/bowling/onions_bowling_2012.ogv",
    "Ben Stokes bowling, 2012.ogv": "motion_videos/bowling/stokes_bowling_2012.ogv",
    "Stephen Parry bowling, 2012.ogv": "motion_videos/bowling/parry_bowling_2012.ogv",
    # Motion: batting
    "Aythorpe Roding CC v Terling CC, Essex, England Video 2 batsman hits cover drives.webm":
        "motion_videos/batting/club_cover_drives.webm",
    "Hatfield Heath CC v Potter Street & Church Langley CC Video 2 pull shot four runs.webm":
        "motion_videos/batting/club_pull_shot.webm",
    "Amar Kala Batting Video 2025.webm": "motion_videos/batting/amar_kala_batting_2025.webm",
}


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def info(titles):
    d = api({"action": "query", "titles": "|".join("File:" + t for t in titles), "prop": "imageinfo",
             "iiprop": "url|size|extmetadata", "iiextmetadatafilter": "LicenseShortName|Artist|LicenseUrl"})
    out = {}
    norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
    for p in d["query"]["pages"].values():
        title = norm.get(p["title"], p["title"])[5:]
        ii = p["imageinfo"][0]
        m = ii.get("extmetadata", {})
        out[title] = {
            "url": ii["url"], "page": ii["descriptionurl"], "size": ii["size"],
            "license": m.get("LicenseShortName", {}).get("value", "?"),
            "license_url": m.get("LicenseUrl", {}).get("value", ""),
            "artist": re.sub(r"<[^>]+>", "", m.get("Artist", {}).get("value", "")).strip(),
        }
    return out


def download(url, dest, tries=6):
    """curl first (Wikimedia throttled urllib's requests with HTTP 429 while curl got through), urllib fallback."""
    import shutil
    import subprocess
    url = url.split("?")[0]
    if shutil.which("curl"):
        for k in range(tries):
            r = subprocess.run(["curl", "-sS", "-f", "-L", "-A", UA, "-o", str(dest), url])
            if r.returncode == 0:
                return
            print(f"  curl failed ({r.returncode}), retry in {20 * (k + 1)}s")
            time.sleep(20 * (k + 1))
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 20 * (k + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)


def main():
    meta = info(list(FILES))
    rows = []
    for title, rel in FILES.items():
        m = meta.get(title) or meta.get(title.replace("_", " "))
        if m is None:
            print("not found on Commons:", title, file=sys.stderr)
            continue
        dest = ROOT / "data" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != m["size"]:
            print(f"downloading {rel} ({m['size'] // 1024} KB)")
            download(m["url"], dest)
            time.sleep(3)  # Wikimedia rate-limits bulk downloads (HTTP 429)
        rows.append((rel, title, m))
    lines = ["# Sources: Wikimedia Commons media in data/", "",
             "Downloaded by `tools/fetch_references.py`. Public domain or CC licences; CC BY / CC BY-SA need",
             "attribution (author + licence + link) wherever these files or derived videos are shown.", "",
             "| file | Commons title | author | licence |", "|---|---|---|---|"]
    for rel, title, m in rows:
        lic = f"[{m['license']}]({m['license_url']})" if m["license_url"] else m["license"]
        lines.append(f"| `data/{rel}` | [{title}]({m['page']}) | {m['artist'] or 'unknown'} | {lic} |")
    (ROOT / "data" / "SOURCES_commons.md").write_text("\n".join(lines) + "\n")
    print("wrote data/SOURCES_commons.md with", len(rows), "files")


if __name__ == "__main__":
    main()
