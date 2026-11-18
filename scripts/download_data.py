"""Download and pack self-contained datasets and character knowledge resources."""

from __future__ import annotations

import gzip
import json
import pickle
import random
import re
import subprocess
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict

DATASET_URL = "https://raw.githubusercontent.com/Red-Scarff/GCC-Spam/HEAD/data/dataset.txt"
SMS_DATASET_URL = "https://raw.githubusercontent.com/hrwhisper/SpamMessage/master/data/%E5%B8%A6%E6%A0%87%E7%AD%BE%E7%9F%AD%E4%BF%A1.txt"
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

PURE_RADICALS_SET = set(
    "氵亻讠扌艹辶宀广疒纟钅阝卩刂忄犭衤礻灬爫耂虍虫页鸟鱼"
    "⺖⺘⺡⺢⺣⺨⺩⺫⺬⺭⺮⺰⺳⺶⺷⺸⺼⺾⻊⻌⻏⻖⻗⻞⻣"
)
COMMON_STOP_CHARS = set("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严")


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
        records.append(f"{lbl}\t{variant.replace(chr(9), ' ')}\t{clean.replace(chr(9), ' ')}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write("\n".join(records) + "\n")
    return len(records)


def build_sms_matched_dataset(base_dataset_path: Path, out_path: Path, n_ham: int = 5500, seed: int = 42) -> int:
    spam_rows, spam_lens = [], []
    with gzip.open(base_dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            parts = s.split("\t")
            if parts[0] == "1":
                spam_rows.append(s)
                spam_lens.append(len(parts[1]))

    raw_sms = fetch_text(SMS_DATASET_URL)
    pua_re = re.compile(r"[\ue000-\uf8ff]")
    pool_by_bin: Dict[int, list] = {}
    seen_ham = set()
    for line in raw_sms.splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) != 2 or parts[0].strip() != "0":
            continue
        txt = re.sub(r"\s+", " ", pua_re.sub("", parts[1]).strip())
        L = len(txt)
        if 15 <= L <= 140 and txt not in seen_ham:
            seen_ham.add(txt)
            pool_by_bin.setdefault(L // 5, []).append(txt)

    rng = random.Random(seed)
    for b in pool_by_bin:
        rng.shuffle(pool_by_bin[b])

    spam_bin_counts = Counter(min(max(L // 5, 3), 27) for L in spam_lens)
    total_s = sum(spam_bin_counts.values())
    selected_hams, used_set = [], set()
    for b, cnt in sorted(spam_bin_counts.items()):
        target_n = max(1, int(round(n_ham * cnt / total_s)))
        for h in pool_by_bin.get(b, []):
            if h not in used_set:
                used_set.add(h)
                selected_hams.append(h)
                if len(selected_hams) >= n_ham or sum(1 for _ in [1]) and target_n <= 1:
                    pass
                target_n -= 1
                if target_n <= 0:
                    break
    if len(selected_hams) < n_ham:
        extra = [h for b in range(7, 18) for h in pool_by_bin.get(b, []) if h not in used_set]
        selected_hams.extend(extra[: n_ham - len(selected_hams)])
    selected_hams = selected_hams[:n_ham]

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for h in selected_hams:
            f.write(f"0\t{h}\t{h}\n")
        for s in spam_rows:
            f.write(s + "\n")
    return len(selected_hams) + len(spam_rows)


def build_chaizi_map(out_path: Path) -> Dict[str, object]:
    pair_to_char: Dict[str, str] = {}
    triple_to_char: Dict[str, str] = {}
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
                elif len(toks) == 3 and all(len(t) == 1 for t in toks):
                    # Guard 3-char splits: require at least one radical & avoid merging common prose words
                    has_rad = any((t in PURE_RADICALS_SET) or ("\u2e80" <= t <= "\u2fdf") for t in toks)
                    no_stop = not any(t in COMMON_STOP_CHARS for t in toks)
                    if has_rad and no_stop:
                        tri = toks[0] + toks[1] + toks[2]
                        if tri not in triple_to_char:
                            triple_to_char[tri] = ch
            if rads:
                existing = set(char_to_radicals.get(ch, []))
                char_to_radicals[ch] = sorted(existing | rads)

    payload = {
        "pair_to_char": pair_to_char,
        "triple_to_char": triple_to_char,
        "manual_splits": MANUAL_SPLIT_MAP,
        "char_to_radicals": char_to_radicals,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
    return payload


def build_ssc_lookup(out_path: Path) -> Dict[str, int]:
    strokes: Dict[str, int] = {}
    for line in fetch_text(UNICODE_TABLE_URL).splitlines()[6:]:
        cols = line.strip().split()
        if len(cols) >= 7 and len(cols[0]) == 1:
            try:
                strokes[cols[0]] = int(cols[6])
            except ValueError:
                pass

    structures: Dict[str, str] = {}
    for line in fetch_text(STRUCTURE_URL).splitlines():
        cols = line.strip().split("\t")
        if len(cols) == 2 and len(cols[1]) == 1:
            structures[cols[1]] = cols[0]

    four_corner: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pip", "download", "-q", "--no-deps", "-d", td, "four-corner-method"], check=True)
        for arch in Path(td).glob("*.whl"):
            with zipfile.ZipFile(arch) as zf:
                for name in zf.namelist():
                    if name.endswith("data.pkl"):
                        fc_data = pickle.loads(zf.read(name))
                        four_corner = {
                            k: str(v) for k, v in fc_data.items()
                            if isinstance(k, str) and len(k) == 1 and not (0xD800 <= ord(k) <= 0xDFFF)
                        }
                        break

    payload = {"strokes": strokes, "structures": structures, "four_corner": four_corner}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
    return {"strokes": len(strokes), "structures": len(structures), "four_corner": len(four_corner)}


def main() -> None:
    res_dir = Path("data/resources")
    n_a = build_variant_dataset(res_dir / "variant_spam_16k.tsv.gz")
    n_b = build_sms_matched_dataset(res_dir / "variant_spam_16k.tsv.gz", res_dir / "variant_spam_sms_matched.tsv.gz")
    chaizi = build_chaizi_map(res_dir / "chaizi_map.json.gz")
    ssc_stats = build_ssc_lookup(res_dir / "ssc_lookup.json.gz")
    print(f"Benchmark A rows: {n_a}, Benchmark B rows: {n_b}", flush=True)
    print(f"Chaizi 2-char: {len(chaizi['pair_to_char'])}, 3-char: {len(chaizi['triple_to_char'])}, chars: {len(chaizi['char_to_radicals'])}", flush=True)
    print(f"SSC stats: {ssc_stats}", flush=True)


if __name__ == "__main__":
    main()
