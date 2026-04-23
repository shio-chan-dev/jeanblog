#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path

CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\$[^$]*\$")
BLOCK_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
FRONT_MATTER_RE = re.compile(r"^---\n.*?\n---\n", re.DOTALL)

CN_RE = re.compile(r"[\u4e00-\u9fff]")
EN_WORD_RE = re.compile(r"[A-Za-z0-9]+")

DEFAULTS = {
    "cn_cpm": 560,
    "en_wpm": 470,
    "code_wpm": 150,
    "math_tokens": 12,
}


def strip_front_matter(text):
    return FRONT_MATTER_RE.sub("", text, count=1)


def extract_code_blocks(text):
    return CODE_FENCE_RE.findall(text)


def remove_code_blocks(text):
    return CODE_FENCE_RE.sub("", text)


def count_cn_chars(text):
    return len(CN_RE.findall(text))


def count_en_words(text):
    return len(EN_WORD_RE.findall(text))


def count_math_blocks(text):
    return len(BLOCK_MATH_RE.findall(text))


def estimate_minutes(text, cn_cpm, en_wpm, code_wpm, math_tokens):
    text = strip_front_matter(text)
    code_blocks = extract_code_blocks(text)
    code_text = " ".join(code_blocks)
    text_wo_code = remove_code_blocks(text)

    math_blocks = count_math_blocks(text_wo_code)
    text_wo_code = BLOCK_MATH_RE.sub("", text_wo_code)
    text_wo_code = INLINE_MATH_RE.sub("", text_wo_code)

    cn_chars = count_cn_chars(text_wo_code)
    en_words = count_en_words(text_wo_code)
    code_words = count_en_words(code_text)

    minutes = math.ceil(
        (cn_chars / cn_cpm)
        + (en_words / en_wpm)
        + (code_words / code_wpm)
        + (math_blocks * (math_tokens / en_wpm))
    )

    return {
        "minutes": minutes,
        "cn_chars": cn_chars,
        "en_words": en_words,
        "code_words": code_words,
        "math_blocks": math_blocks,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate reading time for a markdown file.")
    parser.add_argument("path", type=Path, help="Markdown file path")
    parser.add_argument("--cn-cpm", type=int, default=DEFAULTS["cn_cpm"])
    parser.add_argument("--en-wpm", type=int, default=DEFAULTS["en_wpm"])
    parser.add_argument("--code-wpm", type=int, default=DEFAULTS["code_wpm"])
    parser.add_argument("--math-tokens", type=int, default=DEFAULTS["math_tokens"])
    args = parser.parse_args()

    text = args.path.read_text(encoding="utf-8")
    result = estimate_minutes(text, args.cn_cpm, args.en_wpm, args.code_wpm, args.math_tokens)

    print(
        f"minutes={result['minutes']} cn_chars={result['cn_chars']} "
        f"en_words={result['en_words']} code_words={result['code_words']} "
        f"math_blocks={result['math_blocks']}"
    )


if __name__ == "__main__":
    main()
