#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["wordfreq==3.1.1"]
# ///
"""Write the common English words a session term must not be: src/scribe/data/common_en.txt.gz.

A common word is alphabetic, at least 4 letters, at Zipf frequency 3.0 or more.
The output is byte-reproducible: sorted, and gzipped with a fixed mtime.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path

from wordfreq import iter_wordlist, zipf_frequency

# On 60 measured Claude Code sessions 3.0 admitted 23 names and 1 other word;
# 2.5 lost 6 of the names.
ZIPF = 3.0
MIN_LETTERS = 4
OUT = Path(__file__).resolve().parent.parent / "src" / "scribe" / "data" / "common_en.txt.gz"


def main() -> None:
    """Write the list and print how many words it holds."""
    words: set[str] = set()
    for word in iter_wordlist("en"):
        if word.isalpha() and len(word) >= MIN_LETTERS and zipf_frequency(word, "en") >= ZIPF:
            words.add(word)
    text = "".join(f"{word}\n" for word in sorted(words))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # GzipFile writes the header itself; gzip.compress leaves its OS byte to zlib on 3.12.
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as handle:
        handle.write(text.encode())
    OUT.write_bytes(buffer.getvalue())
    print(len(words))


if __name__ == "__main__":
    main()
