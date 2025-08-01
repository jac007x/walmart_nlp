#!/usr/bin/env python
"""
End-to-end Walmart BU comment processing pipeline.
Run:
    python -m scripts.run_pipeline --csv data/comments_raw.csv
"""
import pandas as pd
import numpy as np
import yaml
import json
import hashlib
import pickle
import re
import spacy
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from scripts.sentiment_ensemble import sentiment_dataframe

tqdm.pandas()
CFG = yaml.safe_load(open("config.yaml"))

# ----------------------------------------------------------------------- #
# 1. Load dictionaries + matcher
theme_dict = json.load(open(CFG["paths"]["theme_dict_core"]))
flag_core  = json.load(open(CFG["paths"]["flag_dict_core"]))
flag_ext   = json.load(open(CFG["paths"]["flag_dict_ext"]))

phrase_matcher, regex_catalogue = pickle.load(
    open(CFG["paths"]["matcher_cache"], "rb")
)

nlp = spacy.load("en_core_web_sm", disable=["ner"])

def negated(tok):
    return any(c.dep_ == "neg" for c in tok.children) or tok.head.dep_ == "neg"

# ----------------------------------------------------------------------- #
# 2. Flag matching
def match_flags(txt, shadow=False):
    lc = txt.lower()
    res = []
    for flag, kws in flag_core.items():
        if any(kw in lc for kw in kws):
            res.append(flag)
    if shadow:
        for flag, kws in flag_ext.items():
            if isinstance(kws, dict):
                if re.search(kws["require_co"], lc) and any(kw in lc for kw in kws["keywords"]):
                    res.append(f"{flag}(shadow)")
            elif any(kw in lc for kw in kws):
                res.append(f"{flag}(shadow)")
    return res

# ----------------------------------------------------------------------- #
# 3. Rule-based theme matching
def match_themes(txt):
    doc = nlp(txt)
    lc  = txt.lower()
    hits, seen = [], set()

    # PhraseMatcher hits
    for mid, start, end in phrase_matcher(doc):
        span = doc[start:end]
        theme = doc.vocab.strings[mid]
        if theme in seen or negated(span.root):
            continue
        cfg = theme_dict.get(theme, {})
        if any(neg in lc for neg in cfg.get("negative_keywords", [])):
            continue
        hits.append(theme)
        seen.add(theme)

    # Regex‐based hits
    for theme, rx in regex_catalogue:
        if theme in seen:
            continue
        if rx.search(txt):
            cfg = theme_dict.get(theme, {})
            if any(neg in lc for neg in cfg.get("negative_keywords", [])):
                continue
            hits.append(theme)
            seen.add(theme)

    return hits

# ----------------------------------------------------------------------- #
# 4. Embedding-assisted fallback (optional)
theme_vecs = {}
if CFG.get("embedding_fallback", {}).get("enabled", False):
    embed_model = SentenceTransformer(CFG["embedding_fallback"]["model"])
    for theme_name, theme_cfg in theme_dict.items():
        exemplars = theme_cfg.get("keywords", [])[:5]
        if not exemplars:
            continue
        emb = embed_model.encode(exemplars, convert_to_tensor=True)
        theme_vecs[theme_name] = emb.mean(dim=0)

def embed_fallback(txt, need):
    if not need:
        return []
    vec  = embed_model.encode([txt], convert_to_tensor=True)
    sims = util.cos_sim(vec, torch.stack(list(theme_vecs.values())))[0]
    th   = CFG["embedding_fallback"]["threshold"]
    idxs = (sims >= th).nonzero(as_tuple=True)[0]
    return [list(theme_vecs.keys())[i] for i in idxs]

# ----------------------------------------------------------------------- #
# 5. Pipeline entrypoint
def run(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    df["CleanComment"] = df["Comment"].fillna("").str.strip()

    # Lineage
    df["Row_SHA1"] = df["CleanComment"].apply(
        lambda t: hashlib.sha1(t.encode()).hexdigest()
    )

    # Flags
    df["Flags_Matched"] = df["CleanComment"].progress_apply(
        match_flags, shadow=True
    )

    # Themes
    df["Themes_Matched"] = df["CleanComment"].progress_apply(match_themes)

    # Embedding fallback
    if CFG.get("embedding_fallback", {}).get("enabled", False):
        need = df["Themes_Matched"].apply(len) == 0
        df.loc[need, "Themes_Matched"] = df.loc[need, "CleanComment"].apply(
            lambda t: embed_fallback(t, True)
        )

    # Sentiment
    df = sentiment_dataframe(df)

    # Save
    out = Path(csv_path).with_suffix(".processed.csv")
    df.to_csv(out, index=False)
    print("Saved →", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    run(args.csv)
