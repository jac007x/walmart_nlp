# /scripts/suggest_terms.py
import pandas as pd, re, yaml, json
from collections import Counter
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv("latest_comments.csv")
unmatched = df[df['Themes_Matched'].str.len()==0]['CleanComment']

# 1. N‑gram mining
tokens = (unmatched
          .str.lower()
          .str.replace(r'[^a-z0-9 ]','', regex=True)
          .str.split())
bigrams = Counter([" ".join(t[i:i+2]) for t in tokens for i in range(len(t)-1)])
candidates = [b for b,c in bigrams.items() if c>=5]

# 2. Embedding similarity to existing themes
model = SentenceTransformer('all-MiniLM-L6-v2')
embeds = model.encode(candidates, convert_to_tensor=True)
theme_centroids = {t: model.encode(" ".join(cfg["keywords"][:4]), convert_to_tensor=True)
                   for t,cfg in json.load(open("dicts/theme_dict_core.json")).items()}
suggest = []
for cand, vec in zip(candidates, embeds):
    sims = {t: util.cos_sim(vec, cent)[0] for t,cent in theme_centroids.items()}
    best_t, score = max(sims.items(), key=lambda kv: kv[1])
    if score>0.55:
        suggest.append((cand,best_t,float(score)))
pd.DataFrame(suggest, columns=["term","suggested_theme","cos_sim"]).to_csv("term_suggestions.csv", index=False)
