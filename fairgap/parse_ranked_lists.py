"""
python3 fairgap/6parse_ranked_lists.py \
  --inp fairgap_out/generations.jsonl \
  --out fairgap_out/ranked_lists.jsonl \
  --top_k 10

[OK] Read 57 generation rows from: fairgap_out/generations.jsonl
[OK] Wrote 57 ranked-list rows to: fairgap_out/ranked_lists.jsonl
[OK] parse_ok == True for 52/57 rows
[OK] Skipped rows: missing_fields=0, bad_user_id=0, bad_variant=0

"""
import argparse
import json
import os
import re
from typing import Any, Dict, Iterator, List, Set


IN_DEFAULT = "examples/toy_out/generations.jsonl"
OUT_DEFAULT = "examples/toy_out/ranked_lists.jsonl"


_PREFIX_RE = re.compile(
    r"^\s*(?:[\-\*\u2022]\s+|(?:\d{1,2})\s*[\.\)\:\-]\s+)\s*"
)

_TITLE_COLON_RE = re.compile(r"^\s*(?:title|movie)\s*:\s*", re.IGNORECASE)

_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"recommended movies|recommendations|top\s*10|top ten|movie recommendations|"
    r"here are|sure[,!:]*|certainly[,!:]*|absolutely[,!:]*|"
    r"the user might like|the user would like|i recommend|my recommendations"
    r")\s*$",
    re.IGNORECASE,
)

_META_RE = re.compile(
    r"^\s*(?:"
    r"output format requirements|strict|exactly 10 lines|one title per line|"
    r"no numbering|no bullets|no extra text|no blank lines"
    r")\s*$",
    re.IGNORECASE,
)

_SURROUND_RE = re.compile(r'^[\s"\']+|[\s"\']+$')


def ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {lineno} in {path}: {e}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object at line {lineno} in {path}, got {type(obj).__name__}"
                )
            yield obj


def normalize_candidate_line(line: str) -> str:
    s = line.strip()
    if not s:
        return ""

    s = _PREFIX_RE.sub("", s)
    s = _TITLE_COLON_RE.sub("", s)
    s = s.strip()
    s = s.rstrip(" \t\r\n-–—•*")
    s = s.strip()
    s = _SURROUND_RE.sub("", s).strip()
    return s


def looks_like_non_title(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _HEADING_RE.match(s):
        return True
    if _META_RE.match(s):
        return True
    return False


def split_output_into_candidate_lines(text: str) -> List[str]:
    raw_lines = text.splitlines()
    nonempty = [x for x in raw_lines if x.strip()]

    if len(nonempty) <= 2 and ";" in text:
        parts: List[str] = []
        for chunk in nonempty if nonempty else [text]:
            parts.extend(chunk.split(";"))
        return [p.strip() for p in parts if p.strip()]

    return [x.strip() for x in raw_lines if x.strip()]


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        key = x.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def parse_ranked_titles(text: str, top_k: int) -> List[str]:
    if not text or not text.strip():
        return []

    candidates = split_output_into_candidate_lines(text)

    parsed: List[str] = []
    for line in candidates:
        if looks_like_non_title(line):
            continue

        s = normalize_candidate_line(line)
        if not s:
            continue

        lowered = s.casefold()
        if lowered.startswith("here are"):
            continue
        if lowered.startswith("recommend"):
            continue
        if lowered.startswith("the user"):
            continue
        if lowered.startswith("based on"):
            continue
        if "i recommend" in lowered:
            continue

        parsed.append(s)

    parsed = dedupe_preserve_order(parsed)
    return parsed[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=IN_DEFAULT, help="Input generations.jsonl")
    ap.add_argument("--out", default=OUT_DEFAULT, help="Output ranked_lists.jsonl")
    ap.add_argument("--top_k", type=int, default=10, help="Keep top-K parsed titles")
    args = ap.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")

    if not os.path.isfile(args.inp):
        raise FileNotFoundError(f"Input file not found: {args.inp}")

    ensure_parent_dir(args.out)

    n_read = 0
    n_written = 0
    n_parse_ok = 0
    n_skipped_missing_fields = 0
    n_skipped_bad_user_id = 0
    n_skipped_bad_variant = 0

    with open(args.out, "w", encoding="utf-8") as f_out:
        for row in iter_jsonl(args.inp):
            n_read += 1

            if "user_id" not in row or "variant" not in row:
                n_skipped_missing_fields += 1
                continue

            try:
                user_id = int(row["user_id"])
            except Exception:
                n_skipped_bad_user_id += 1
                continue

            variant = str(row.get("variant", "")).strip()
            if variant not in ("age_a", "age_b"):
                n_skipped_bad_variant += 1
                continue

            raw_output_text = str(row.get("output_text", "") or "")

            pair_metadata = row.get("pair_metadata", {})
            if not isinstance(pair_metadata, dict):
                pair_metadata = {}

            ranked_titles = parse_ranked_titles(raw_output_text, top_k=args.top_k)
            n_titles = len(ranked_titles)
            parse_ok = (n_titles == args.top_k)

            rec = {
                "user_id": user_id,
                "variant": variant,
                "ranked_titles": ranked_titles,
                "n_titles_parsed": n_titles,
                "parse_ok": bool(parse_ok),
                "raw_output_text": raw_output_text,
                "pair_metadata": pair_metadata,
            }

            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

            if parse_ok:
                n_parse_ok += 1

    print(f"[OK] Read {n_read} generation rows from: {args.inp}")
    print(f"[OK] Wrote {n_written} ranked-list rows to: {args.out}")
    print(f"[OK] parse_ok == True for {n_parse_ok}/{n_written} rows")
    print(
        "[OK] Skipped rows: "
        f"missing_fields={n_skipped_missing_fields}, "
        f"bad_user_id={n_skipped_bad_user_id}, "
        f"bad_variant={n_skipped_bad_variant}"
    )


if __name__ == "__main__":
    main()