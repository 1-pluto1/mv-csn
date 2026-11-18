"""Download and pack self-contained datasets and character knowledge resources."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path
from typing import Dict

DATASET_URL = "https://raw.githubusercontent.com/Red-Scarff/GCC-Spam/HEAD/data/dataset.txt"
CHAIZI_JT_URL = "https://raw.githubusercontent.com/kfcd/chaizi/master/chaizi-jt.txt"
CHAIZI_FT_URL = "https://raw.githubusercontent.com/kfcd/chaizi/master/chaizi-ft.txt"
UNICODE_TABLE_URL = "https://raw.githubusercontent.com/Red-Scarff/GCC-Spam/HEAD/data/chinese_unicode_table.txt"
STRUCTURE_URL = "https://raw.githubusercontent.com/Red-Scarff/GCC-Spam/HEAD/data/hanzijiegou_2w.txt"

MANUAL_SPLIT_MAP: Dict[str, str] = {
    "\\/": "v",
    "|/": "v",
    "十Ꝗ": "加Q",
    "十Q": "加Q",
    "十q": "加q",
    "c0n": "com",
    "c0m": "com",
    "C0M": "COM",
}


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def build_variant_dataset(out_path: Path) -> int:
    raw = fetch_text(DATASET_URL)
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        lbl = parts[0].strip()
        if lbl not in ("0", "1"):
            continue
        variant = parts[1].strip() if len(parts) >= 2 else ""
        clean = parts[2].strip() if len(parts) >= 3 else variant
        if not variant:
            continue
        variant = variant.replace("\t", " ")
        clean = clean.replace("\t", " ")
        records.append(f"{lbl}\t{variant}\t{clean}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write("\n".join(records) + "\n")
    return len(records)


def build_chaizi_map(out_path: Path) -> Dict[str, object]:
    pair_to_char: Dict[str, str] = {}
    char_to_radicals: Dict[str, list] = {}
    for url in (CHAIZI_JT_URL, CHAIZI_FT_URL):
        text = fetch_text(url)
        for line in text.splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            ch = parts[0].strip()
            if len(ch) != 1:
                continue
            rads = set()
            for decomp in parts[1:]:
                toks = [t for t in decomp.strip().split() if t]
                for t in toks:
                    rads.add(t)
                if len(toks) == 2 and all(len(t) == 1 for t in toks):
                    pair = toks[0] + toks[1]
                    if pair not in pair_to_char:
                        pair_to_char[pair] = ch
            if rads:
                existing = set(char_to_radicals.get(ch, []))
                char_to_radicals[ch] = sorted(existing | rads)

    payload = {
        "pair_to_char": pair_to_char,
        "manual_splits": MANUAL_SPLIT_MAP,
        "char_to_radicals": char_to_radicals,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def build_ssc_lookup(out_path: Path) -> Dict[str, int]:
    strokes: Dict[str, int] = {}
    uni_text = fetch_text(UNICODE_TABLE_URL)
    for line in uni_text.splitlines()[6:]:
        cols = line.strip().split()
        if len(cols) >= 7 and len(cols[0]) == 1:
            try:
                strokes[cols[0]] = int(cols[6])
            except ValueError:
                pass

    structures: Dict[str, str] = {}
    struct_text = fetch_text(STRUCTURE_URL)
    for line in struct_text.splitlines():
        cols = line.strip().split("\t")
        if len(cols) == 2 and len(cols[1]) == 1:
            structures[cols[1]] = cols[0]

    payload = {
        "strokes": strokes,
        "structures": structures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return {"strokes": len(strokes), "structures": len(structures)}


def main() -> None:
    res_dir = Path("data/resources")
    n_rows = build_variant_dataset(res_dir / "variant_spam_16k.tsv.gz")
    chaizi = build_chaizi_map(res_dir / "chaizi_map.json.gz")
    ssc_stats = build_ssc_lookup(res_dir / "ssc_lookup.json.gz")
    print(f"Dataset saved: {n_rows} rows", flush=True)
    print(f"Chaizi pairs: {len(chaizi['pair_to_char'])}, chars with radicals: {len(chaizi['char_to_radicals'])}", flush=True)
    print(f"SSC lookup stats: {ssc_stats}", flush=True)


if __name__ == "__main__":
    main()
