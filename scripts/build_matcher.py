"""
Compile spaCy PhraseMatcher + regex catalogue and cache to disk.
Run after every dictionary change.
"""
import json, pickle, re, spacy, pathlib, yaml
from spacy.matcher import PhraseMatcher
import yaml

CFG = yaml.safe_load(open("config.yaml"))
THEME = json.load(open(CFG["paths"]["theme_dict_core"]))

nlp = spacy.load("en_core_web_sm", disable=["ner","parser"])
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
regex_catalogue = []

for theme, cfg in THEME.items():
    for kw in cfg.get("keywords", []):
        if isinstance(kw, dict) and "__regex__" in kw:
            regex_catalogue.append((theme, re.compile(kw["__regex__"], re.I)))
        else:
            matcher.add(theme, [nlp.make_doc(kw)])

pickle.dump((matcher, regex_catalogue), open(CFG["paths"]["matcher_cache"], "wb"))
print("Matcher cache compiled.")
