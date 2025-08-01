"""
Run once to canonicalise & deduplicate the original theme dictionary.
Usage:
    python scripts/normalise_theme_dict.py \
        --in  dicts/walmart_theme_dict_original.json \
        --out dicts/theme_dict_core.json
"""
import json, argparse, unicodedata, spacy, pathlib, re
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

def lemma(s: str) -> str:
    doc = nlp(unicodedata.normalize("NFKD", s))
    return " ".join(t.lemma_.lower() for t in doc)

def dedupe(node: dict) -> dict:
    for key in ("keywords", "negative_keywords"):
        if key in node:
            node[key] = sorted({lemma(k) for k in node[key]})
    for sub, cfg in node.get("subthemes", {}).items():
        if isinstance(cfg, dict):
            node["subthemes"][sub] = dedupe(cfg)
        else:
            node["subthemes"][sub] = sorted({lemma(k) for k in cfg})
    return node

def main(src, dst):
    core = json.load(open(src))
    cleaned = {t: dedupe(cfg) for t, cfg in core.items()}
    pathlib.Path(dst).write_text(json.dumps(cleaned, indent=2, ensure_ascii=False))
    print(f"Saved normalised dict → {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args().in, ap.parse_args().out)
