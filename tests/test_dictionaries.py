import json, pytest, yaml, re, pickle, spacy
CFG   = yaml.safe_load(open("config.yaml"))
THEME = json.load(open(CFG["paths"]["theme_dict_core"]))

def test_unique_keywords():
    seen = set()
    for theme, cfg in THEME.items():
        for kw in cfg["keywords"]:
            assert kw not in seen, f"{kw} duplicated"
            seen.add(kw)

def test_phrase_matcher_loads():
    pickle.load(open(CFG["paths"]["matcher_cache"], "rb"))
