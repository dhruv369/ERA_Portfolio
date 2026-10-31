#!/usr/bin/env python3
"""Reproducible, bounded Samanantar text-cleaning pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import re
import sqlite3
import struct
import sys
import unicodedata
import platform
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


DATASET = "ai4bharat/samanantar"
REVISION = "ead34c5b22a0354d350e7dfdd6aece7df5a48f37"
LANGUAGES = ("as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te")
MAX_INPUT_TOKENS = 10_000_000
STAGE_ORDER = (
    "boilerplate",
    "unicode",
    "special_tags",
    "language_id",
    "quality",
    "decontamination",
    "pii",
    "minhash_lsh",
)

BOILERPLATE_PATTERNS = (
    re.compile(r"\b(?:enable|accept) cookies to continue\b", re.I),
    re.compile(r"\bsubscribe to (?:continue|read)\b", re.I),
    re.compile(r"\bthis page requires javascript\b", re.I),
    re.compile(r"^\s*(?:cookie policy|privacy policy)\s*$", re.I),
)
GHOST_TAG_PATTERN = re.compile(
    r"\[(?:/?(?:USER|SYSTEM|ASSISTANT|INST|PROMPT|RESPONSE))\]"
    r"|<\|(?:endoftext|im_start|im_sep|fim_prefix|fim_middle|fim_suffix)\|>"
    r"|<extra_id_\d+>",
    re.I,
)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d().\-\s]{7,}\d)(?!\w)")
WORD_PATTERN = re.compile(r"\S+")

INDIC_RANGES = {
    "as": ((0x0980, 0x09FF),),
    "bn": ((0x0980, 0x09FF),),
    "gu": ((0x0A80, 0x0AFF),),
    "hi": ((0x0900, 0x097F),),
    "kn": ((0x0C80, 0x0CFF),),
    "ml": ((0x0D00, 0x0D7F),),
    "mr": ((0x0900, 0x097F),),
    "or": ((0x0B00, 0x0B7F),),
    "pa": ((0x0A00, 0x0A7F),),
    "ta": ((0x0B80, 0x0BFF),),
    "te": ((0x0C00, 0x0C7F),),
}
MINHASH_PRIME = (1 << 61) - 1
MINHASH_COEFFICIENTS = tuple(
    (
        int.from_bytes(
            hashlib.blake2b(
                b"s4-minhash-v1" + seed.to_bytes(4, "big"), digest_size=16
            ).digest()[:8],
            "big",
        )
        % (MINHASH_PRIME - 1)
        + 1,
        int.from_bytes(
            hashlib.blake2b(
                b"s4-minhash-v1" + seed.to_bytes(4, "big"), digest_size=16
            ).digest()[8:],
            "big",
        )
        % MINHASH_PRIME,
    )
    for seed in range(128)
)


def word_tokens(text: str) -> list[str]:
    """The project's explicit token unit: maximal non-whitespace runs."""
    return WORD_PATTERN.findall(text)


def record_token_count(row: dict[str, Any]) -> int:
    return len(word_tokens(row["src"])) + len(word_tokens(row["tgt"]))


def _contains_boilerplate(text: str) -> bool:
    return any(pattern.search(text) for pattern in BOILERPLATE_PATTERNS)


def _script_mismatch(text: str, language: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    ranges = INDIC_RANGES.get(language, ())
    expected = sum(
        any(start <= ord(char) <= end for start, end in ranges) for char in letters
    )
    return expected / len(letters) < 0.25


def _quality_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for side in ("src", "tgt"):
        text = row[side]
        tokens = word_tokens(text)
        if len(text.strip()) < 5:
            reasons.append(f"{side}_under_5_chars")
        if len(text) > 2000:
            reasons.append(f"{side}_over_2000_chars")
        if not tokens or max(map(len, tokens), default=0) > 100:
            reasons.append(f"{side}_extreme_token")
        letters = sum(char.isalpha() for char in text)
        if len(text) >= 20 and letters / len(text) < 0.15:
            reasons.append(f"{side}_low_letter_ratio")
        if re.search(r"(.)\1{9,}", text):
            reasons.append(f"{side}_repeated_character")
    src_words = len(word_tokens(row["src"]))
    tgt_words = len(word_tokens(row["tgt"]))
    if min(src_words, tgt_words) and max(src_words, tgt_words) / min(src_words, tgt_words) > 8:
        reasons.append("extreme_pair_length_ratio")
    return reasons


def _minhash(shingles: set[bytes], permutations: int) -> tuple[int, ...]:
    if not shingles:
        return ()
    hashes = [
        int.from_bytes(hashlib.blake2b(shingle, digest_size=8).digest(), "big")
        % MINHASH_PRIME
        for shingle in shingles
    ]
    signature: list[int] = []
    for multiplier, offset in MINHASH_COEFFICIENTS[:permutations]:
        signature.append(
            min(
                (multiplier * value + offset) % MINHASH_PRIME
                for value in hashes
            )
        )
    return tuple(signature)


def _shingles(row: dict[str, Any], size: int = 5) -> set[bytes]:
    words = word_tokens(f"{row['src']} {row['tgt']}".casefold())
    if not words:
        return set()
    width = min(size, len(words))
    return {
        " ".join(words[index : index + width]).encode("utf-8")
        for index in range(len(words) - width + 1)
    }


class MinHashLSH:
    """SQLite-backed, deterministic MinHash/LSH near-duplicate index."""

    def __init__(
        self,
        path: Path,
        threshold: float = 0.85,
        permutations: int = 128,
        bands: int = 32,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if not 0 < permutations <= len(MINHASH_COEFFICIENTS):
            raise ValueError(f"permutations must be <= {len(MINHASH_COEFFICIENTS)}")
        if permutations % bands:
            raise ValueError("permutations must be divisible by bands")
        self.threshold = threshold
        self.permutations = permutations
        self.bands = bands
        self.rows_per_band = permutations // bands
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS buckets "
            "(band INTEGER NOT NULL, signature BLOB NOT NULL, doc_id INTEGER NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS bucket_lookup ON buckets(band, signature)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS documents (doc_id INTEGER PRIMARY KEY, shingles TEXT)"
        )
        self.next_id = self.connection.execute(
            "SELECT COALESCE(MAX(doc_id), 0) + 1 FROM documents"
        ).fetchone()[0]

    def is_duplicate(self, row: dict[str, Any]) -> bool:
        shingles = _shingles(row)
        if not shingles:
            return False
        signature = _minhash(shingles, self.permutations)
        candidates: set[int] = set()
        for band in range(self.bands):
            start = band * self.rows_per_band
            payload = struct.pack(
                f">{self.rows_per_band}Q",
                *signature[start : start + self.rows_per_band],
            )
            candidates.update(
                doc_id
                for (doc_id,) in self.connection.execute(
                    "SELECT doc_id FROM buckets WHERE band=? AND signature=?",
                    (band, payload),
                )
            )
        for candidate in candidates:
            raw = self.connection.execute(
                "SELECT shingles FROM documents WHERE doc_id=?", (candidate,)
            ).fetchone()[0]
            prior = {item.encode("utf-8") for item in json.loads(raw)}
            if len(shingles & prior) / len(shingles | prior) >= self.threshold:
                return True

        doc_id = self.next_id
        self.next_id += 1
        self.connection.execute(
            "INSERT INTO documents(doc_id, shingles) VALUES (?, ?)",
            (doc_id, json.dumps([item.decode("utf-8") for item in sorted(shingles)])),
        )
        for band in range(self.bands):
            start = band * self.rows_per_band
            payload = struct.pack(
                f">{self.rows_per_band}Q",
                *signature[start : start + self.rows_per_band],
            )
            self.connection.execute(
                "INSERT INTO buckets(band, signature, doc_id) VALUES (?, ?, ?)",
                (band, payload, doc_id),
            )
        if doc_id % 1000 == 0:
            self.connection.commit()
        return False

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class HoldoutMatcher:
    def __init__(self, path: Path) -> None:
        references = [
            " ".join(line.casefold().split())
            for line in path.read_text(encoding="utf-8").splitlines()
            if len(line.strip()) >= 40
        ]
        self.references = references

    def matches(self, row: dict[str, Any]) -> bool:
        for text in (row["src"], row["tgt"]):
            normalized = " ".join(text.casefold().split())
            if len(normalized) < 40:
                continue
            if any(
                normalized in reference or reference in normalized
                for reference in self.references
            ):
                return True
        return False


def process_rows(
    rows: Iterable[dict[str, Any]],
    output_dir: Path,
    target_tokens: int = MAX_INPUT_TOKENS,
    disabled: set[str] | None = None,
    dedup_threshold: float = 0.85,
    holdout_file: Path | None = None,
    dataset_revision: str = REVISION,
    configured_languages: tuple[str, ...] = LANGUAGES,
) -> dict[str, Any]:
    """Clean rows from an iterable; writes a bounded slice and reproducibility stats."""
    if not 0 < target_tokens <= MAX_INPUT_TOKENS:
        raise ValueError(f"target_tokens must be between 1 and {MAX_INPUT_TOKENS:,}")
    disabled = disabled or set()
    unknown = disabled - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(sorted(unknown))}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_dir / "cleaned.jsonl"
    index_path = output_dir / ".minhash-index.sqlite3"
    if index_path.exists():
        index_path.unlink()
    dedup: MinHashLSH | None = None
    holdout = HoldoutMatcher(holdout_file) if holdout_file else None
    metrics: dict[str, dict[str, Any]] = {
        stage: {
            "status": "not_applied" if stage in disabled else "applied",
            "input_documents": 0,
            "output_documents": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "dropped_documents": 0,
            "modified_documents": 0,
            "findings": 0,
        }
        for stage in STAGE_ORDER
    }
    if holdout is None and "decontamination" not in disabled:
        metrics["decontamination"]["status"] = "not_run_no_holdout_file"
    language_counts: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    language_mismatches: Counter[str] = Counter()
    pii_counts: Counter[str] = Counter()
    special_tag_count = 0
    quality_reasons: Counter[str] = Counter()
    input_documents = 0
    input_tokens = 0
    slice_tokens = 0
    output_tokens = 0
    config_tokens: Counter[str] = Counter()
    output_language_counts: Counter[str] = Counter()
    output_language_tokens: Counter[str] = Counter()

    if "minhash_lsh" not in disabled:
        dedup = MinHashLSH(index_path, threshold=dedup_threshold)

    def stage_start(stage: str, count: int, tokens: int) -> None:
        metrics[stage]["input_documents"] += count
        metrics[stage]["input_tokens"] += tokens

    def stage_end(stage: str, count: int, tokens: int) -> None:
        metrics[stage]["output_documents"] += count
        metrics[stage]["output_tokens"] += tokens
        metrics[stage]["dropped_documents"] += int(count == 0)

    try:
        with cleaned_path.open("w", encoding="utf-8") as output:
            for incoming in rows:
                row = {
                    "idx": incoming.get("idx"),
                    "src": str(incoming["src"]),
                    "tgt": str(incoming["tgt"]),
                    "language": str(incoming["language"]),
                    "src_language": "en",
                }
                original_tokens = record_token_count(row)
                if slice_tokens + original_tokens > target_tokens:
                    break
                slice_tokens += original_tokens
                input_documents += 1
                input_tokens += original_tokens
                config_tokens[row["language"]] += original_tokens
                if input_documents % 50_000 == 0:
                    print(
                        f"Read {input_documents:,} pairs / {input_tokens:,} whitespace tokens",
                        file=sys.stderr,
                        flush=True,
                    )
                count, tokens = 1, original_tokens

                stage_start("boilerplate", count, tokens)
                if "boilerplate" not in disabled and (
                    _contains_boilerplate(row["src"]) or _contains_boilerplate(row["tgt"])
                ):
                    stage_end("boilerplate", 0, 0)
                    continue
                stage_end("boilerplate", count, tokens)

                stage_start("unicode", count, tokens)
                unicode_changed = False
                if "unicode" not in disabled:
                    previous = row["src"], row["tgt"]
                    row["src"], row["tgt"] = (
                        unicodedata.normalize("NFC", row["src"]),
                        unicodedata.normalize("NFC", row["tgt"]),
                    )
                    unicode_changed = previous != (row["src"], row["tgt"])
                    metrics["unicode"]["modified_documents"] += int(unicode_changed)
                    metrics["unicode"]["findings"] += int(unicode_changed)
                stage_end("unicode", count, record_token_count(row))

                stage_start("special_tags", count, record_token_count(row))
                tags_changed = False
                if "special_tags" not in disabled:
                    before = row["src"], row["tgt"]
                    matches = sum(
                        len(GHOST_TAG_PATTERN.findall(text)) for text in before
                    )
                    special_tag_count += matches
                    row["src"], row["tgt"] = tuple(
                        GHOST_TAG_PATTERN.sub("", text) for text in before
                    )
                    tags_changed = before != (row["src"], row["tgt"])
                    metrics["special_tags"]["modified_documents"] += int(tags_changed)
                    metrics["special_tags"]["findings"] += matches
                stage_end("special_tags", count, record_token_count(row))

                stage_start("language_id", count, record_token_count(row))
                if "language_id" not in disabled:
                    language_counts[row["language"]] += 1
                    language_tokens[row["language"]] += record_token_count(row)
                    language_mismatches[row["language"]] += int(
                        _script_mismatch(row["tgt"], row["language"])
                    )
                    metrics["language_id"]["findings"] += int(
                        _script_mismatch(row["tgt"], row["language"])
                    )
                stage_end("language_id", count, record_token_count(row))

                stage_start("quality", count, record_token_count(row))
                if "quality" not in disabled:
                    reasons = _quality_reasons(row)
                    quality_reasons.update(reasons)
                    metrics["quality"]["findings"] += len(reasons)
                    if reasons:
                        stage_end("quality", 0, 0)
                        continue
                stage_end("quality", count, record_token_count(row))

                stage_start("decontamination", count, record_token_count(row))
                if holdout and "decontamination" not in disabled and holdout.matches(row):
                    stage_end("decontamination", 0, 0)
                    metrics["decontamination"]["findings"] += 1
                    continue
                if holdout and "decontamination" in disabled:
                    metrics["decontamination"]["status"] = "not_applied"
                stage_end("decontamination", count, record_token_count(row))

                stage_start("pii", count, record_token_count(row))
                pii_changed = False
                if "pii" not in disabled:
                    before = row["src"], row["tgt"]
                    for side in ("src", "tgt"):
                        emails = EMAIL_PATTERN.findall(row[side])
                        phones = PHONE_PATTERN.findall(row[side])
                        pii_counts["email"] += len(emails)
                        pii_counts["phone"] += len(phones)
                        row[side] = EMAIL_PATTERN.sub("[EMAIL]", row[side])
                        row[side] = PHONE_PATTERN.sub("[PHONE]", row[side])
                    pii_changed = before != (row["src"], row["tgt"])
                    metrics["pii"]["findings"] += (
                        len(EMAIL_PATTERN.findall(before[0]))
                        + len(EMAIL_PATTERN.findall(before[1]))
                        + len(PHONE_PATTERN.findall(before[0]))
                        + len(PHONE_PATTERN.findall(before[1]))
                    )
                    metrics["pii"]["modified_documents"] += int(pii_changed)
                stage_end("pii", count, record_token_count(row))

                stage_start("minhash_lsh", count, record_token_count(row))
                if dedup and dedup.is_duplicate(row):
                    stage_end("minhash_lsh", 0, 0)
                    continue
                stage_end("minhash_lsh", count, record_token_count(row))

                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                cleaned_tokens = record_token_count(row)
                output_tokens += cleaned_tokens
                output_language_counts[row["language"]] += 1
                output_language_tokens[row["language"]] += cleaned_tokens
    finally:
        if dedup:
            dedup.close()
        if index_path.exists():
            index_path.unlink()

    stage_parameters = {
        "boilerplate": {
            "patterns": [pattern.pattern for pattern in BOILERPLATE_PATTERNS],
            "action": "drop entire pair on a phrase match",
        },
        "unicode": {
            "form": "NFC",
            "preserve": ["U+200C ZWNJ", "U+200D ZWJ"],
            "whitespace": "unchanged; no whitespace collapse",
            "scope": "src and tgt fields only",
        },
        "special_tags": {
            "patterns": GHOST_TAG_PATTERN.pattern,
            "action": "remove matched ghost/control spellings; preserve joiners",
        },
        "language_id": {
            "method": "dataset config metadata for target; source is English",
            "classifier": "not used; no model-based language-ID dependency",
            "additional_check": "target-script heuristic, mismatch is an audit flag only",
        },
        "quality": {
            "minimum_chars_per_side": 5,
            "maximum_chars_per_side": 2000,
            "maximum_single_whitespace_token_chars": 100,
            "minimum_letter_ratio_for_text_at_least_20_chars": 0.15,
            "maximum_pair_whitespace_token_ratio": 8,
            "repeated_character_limit": 10,
        },
        "pii": {
            "patterns": {"email": EMAIL_PATTERN.pattern, "phone": PHONE_PATTERN.pattern},
            "replacement": "[EMAIL] / [PHONE]",
            "counts": dict(pii_counts),
        },
        "minhash_lsh": {
            "threshold": dedup_threshold,
            "similarity": "exact Jaccard on five-word pair shingles after LSH candidate retrieval",
            "shingle_hash": "64-bit BLAKE2b",
            "permutation": "deterministic affine min-hash over 2^61-1",
            "permutations": 128,
            "bands": 32,
            "rows_per_band": 4,
            "order": "after decontamination and PII replacement",
        },
        "decontamination": {
            "status": metrics["decontamination"]["status"],
            "method": "case-folded whitespace-normalized substring match against supplied holdout lines >=40 chars",
            "holdout_file": str(holdout_file) if holdout_file else None,
            "order": "after heuristic quality filtering; before PII replacement and deduplication",
            "limits": "No benchmark list is bundled; no comprehensive or n-gram overlap claim.",
        },
    }
    stats = {
        "dataset": DATASET,
        "revision": dataset_revision,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": platform.python_version(),
            "datasets": importlib.metadata.version("datasets")
            if importlib.metadata.packages_distributions().get("datasets")
            else None,
        },
        "run_configuration": {
            "disabled_stages": sorted(disabled),
            "dedup_threshold": dedup_threshold,
            "configured_languages": list(configured_languages),
            "stream_order": list(configured_languages),
        },
        "target_tokens": target_tokens,
        "token_definition": "one maximal non-whitespace run, counted in src plus tgt; not a model tokenizer",
        "slice": {
            "input_documents": input_documents,
            "input_tokens": input_tokens,
            "output_documents": sum(output_language_counts.values()),
            "output_tokens": output_tokens,
            "token_budget_note": "input slice stops before the first pair that would exceed target_tokens",
            "languages": dict(config_tokens),
        },
        "language_split": {
            lang: {
                "documents_assigned": language_counts[lang],
                "tokens_assigned": language_tokens[lang],
                "output_documents": output_language_counts[lang],
                "output_tokens": output_language_tokens[lang],
                "script_mismatch_flags": language_mismatches[lang],
            }
            for lang in LANGUAGES
        },
        "stages": metrics,
        "deduplication": {
            "status": metrics["minhash_lsh"]["status"],
            "documents_checked": metrics["minhash_lsh"]["input_documents"],
            "duplicates_removed": metrics["minhash_lsh"]["dropped_documents"],
            "duplicate_rate": (
                metrics["minhash_lsh"]["dropped_documents"]
                / metrics["minhash_lsh"]["input_documents"]
                if metrics["minhash_lsh"]["input_documents"]
                and "minhash_lsh" not in disabled
                else None
            ),
            "rate_definition": "LSH-retrieved rows removed at exact Jaccard >= threshold / rows reaching dedup stage; approximate candidate recall",
        },
        "parameters": stage_parameters,
        "quality_reasons": dict(quality_reasons),
        "special_tag_occurrences_removed": special_tag_count,
        "pipeline_notes": [
            "Input and output token totals count whitespace tokens, not model tokenizer tokens.",
            "Document totals are sentence pairs because Samanantar is parallel sentence-pair data.",
            "Language split is dataset-config metadata; script heuristic is not an independent LID classifier.",
            "Funnel counts use documents surviving each sequential stage; modified-document and finding counts are separate.",
        ],
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def iter_samanantar(
    languages: tuple[str, ...] = LANGUAGES,
    revision: str = REVISION,
) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Fetching requires `pip install -r Assignments/S4/requirements.txt`."
        ) from error

    streams = {
        language: iter(
            load_dataset(
                DATASET,
                language,
                split="train",
                streaming=True,
                revision=revision,
            )
        )
        for language in languages
    }
    active = list(languages)
    while active:
        for language in tuple(active):
            try:
                row = next(streams[language])
            except StopIteration:
                active.remove(language)
                continue
            yield {
                "idx": row.get("idx"),
                "src": row["src"],
                "tgt": row["tgt"],
                "language": language,
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=MAX_INPUT_TOKENS,
        help=f"maximum raw input whitespace tokens (1–{MAX_INPUT_TOKENS:,})",
    )
    parser.add_argument("--languages", default=",".join(LANGUAGES))
    parser.add_argument("--disable", default="", help="Comma-separated stage names")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    parser.add_argument("--holdout-file", type=Path)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args(argv)
    languages = tuple(item.strip() for item in args.languages.split(",") if item.strip())
    unknown_languages = set(languages) - set(LANGUAGES)
    if unknown_languages:
        parser.error(f"Unsupported Samanantar language configs: {sorted(unknown_languages)}")
    disabled = {item.strip() for item in args.disable.split(",") if item.strip()}
    if not 0 < args.target_tokens <= MAX_INPUT_TOKENS:
        parser.error(
            f"--target-tokens must be between 1 and {MAX_INPUT_TOKENS:,}"
        )
    try:
        stats = process_rows(
            iter_samanantar(languages, revision=args.revision),
            args.output_dir,
            target_tokens=args.target_tokens,
            disabled=disabled,
            dedup_threshold=args.dedup_threshold,
            holdout_file=args.holdout_file,
            dataset_revision=args.revision,
            configured_languages=languages,
        )
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(stats["slice"], ensure_ascii=False, indent=2))
    print(f"Stats: {args.output_dir / 'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
