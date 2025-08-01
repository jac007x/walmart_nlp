#!/usr/bin/env python3
"""
Run once to canonicalise & deduplicate the original theme dictionary.

Usage:
    python scripts/normalise_theme_dict.py \
        --src dicts/walmart_theme_dict_original.json \
        --dst dicts/theme_dict_core.json
"""

import json
import argparse
import unicodedata
import spacy
import pathlib

# Load spaCy, but disable components we don't need for speed
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

def lemma(s: str) -> str:
    """
    Normalize and lemmatize a string.
    """
    text = unicodedata.normalize("NFKD", s)
    doc = nlp(text)
    return " ".join(tok.lemma_.lower() for tok in doc if not tok.is_punct)

def dedupe(node: dict) -> dict:
    """
    Recursively dedupe and normalize 'keywords', 'negative_keywords',
    and any subtheme lists.
    """
    for key in ("keywords", "negative_keywords"):
        if key in node:
            node[key] = sorted({lemma(item) for item in node[key]})

    for subname, cfg in node.get("subthemes", {}).items():
        if isinstance(cfg, dict):
            node["subthemes"][subname] = dedupe(cfg)
        else:
            node["subthemes"][subname] = sorted({lemma(item) for item in cfg})

    return node

def main(src_path: str, dst_path: str):
    raw = json.loads(pathlib.Path(src_path).read_text(encoding="utf-8"))
    cleaned = {theme: dedupe(cfg) for theme, cfg in raw.items()}
    out = json.dumps(cleaned, indent=2, ensure_ascii=False)
    pathlib.Path(dst_path).write_text(out, encoding="utf-8")
    print(f"✅ Saved normalised dict → {dst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize and dedupe the Walmart theme dictionary"
    )
    parser.add_argument(
        "--src", "-i",
        dest="src",
        required=True,
        help="Path to raw theme JSON (e.g. dicts/walmart_theme_dict_original.json)"
    )
    parser.add_argument(
        "--dst", "-o",
        dest="dst",
        required=True,
        help="Path where to write normalized JSON (e.g. dicts/theme_dict_core.json)"
    )
    args = parser.parse_args()
    main(args.src, args.dst)
