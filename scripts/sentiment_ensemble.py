import torch
import pandas as pd
import numpy as np
import yaml
import json
import re

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Load config & initialize models
# --------------------------------------------------------------------------- #
CFG = yaml.safe_load(open("config.yaml"))

device = 0 if torch.cuda.is_available() else -1

# VADER for fast rule-based scoring
vader = SentimentIntensityAnalyzer()

# DistilBERT for a second opinion
bert = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    batch_size=CFG["batch_sizes"]["transformer"],
)

# OpenAI client
client = OpenAI()

# --------------------------------------------------------------------------- #
# 1. VADER helper
# --------------------------------------------------------------------------- #
def vader_score(txt: str):
    c = vader.polarity_scores(txt)["compound"]
    if c >= 0.6:
        return "strongly positive", c
    if c >= 0.2:
        return "positive", c
    if c > -0.2:
        return "neutral", c
    if c > -0.6:
        return "negative", abs(c)
    return "strongly negative", abs(c)

# --------------------------------------------------------------------------- #
# 2. BERT batch helper
# --------------------------------------------------------------------------- #
def bert_batch(texts: list[str]):
    out   = bert(texts)
    lbl   = ["positive" if r["label"] == "POSITIVE" else "negative" for r in out]
    score = [r["score"] for r in out]
    return lbl, score

# --------------------------------------------------------------------------- #
# 3. GPT fallback helper (synchronous)
# --------------------------------------------------------------------------- #
def gpt_refine(texts: list[str]) -> tuple[list[str], list[float]]:
    """
    Calls OpenAI synchronously (no 'acreate') to get JSON of
    { "sentiment": ..., "confidence": ... } for each text.
    """
    # system prompt + user messages
    messages = [
        {
            "role": "system",
            "content": (
                "Classify sentiment as one of: strongly positive, positive, "
                "neutral, negative, or strongly negative. Then return JSON "
                'with keys "sentiment" and "confidence" (0.0–1.0).'
            )
        }
    ]
    messages += [{"role": "user", "content": t} for t in texts]

    resp = client.chat.completions.create(
        model      = CFG["gpt"]["model"],
        messages   = messages,
        max_tokens = 30,
    )

    # parse each choice.message.content as JSON
    results = [json.loads(choice.message.content) for choice in resp.choices]
    sentiments = [r["sentiment"]   for r in results]
    confidences= [r["confidence"]  for r in results]
    return sentiments, confidences

# --------------------------------------------------------------------------- #
# 4. Ensemble logic per row
# --------------------------------------------------------------------------- #
def ensemble_row(v_lbl, v_c, b_lbl, b_c):
    """
    If VADER and BERT agree with confidence >=0.4, take that.
    Otherwise mark for GPT.
    """
    if v_lbl == b_lbl and max(v_c, b_c) >= 0.4:
        return v_lbl, max(v_c, b_c), "rule"
    return None, None, "needs_gpt"

# --------------------------------------------------------------------------- #
# 5. Main entrypoint — apply to DataFrame
# --------------------------------------------------------------------------- #
def sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # VADER scores
    vader_res = df["CleanComment"].apply(lambda t: pd.Series(vader_score(t)))
    df["Sent_VADER"], df["Conf_VADER"] = vader_res[0], vader_res[1]

    # BERT scores
    b_lbl, b_c = bert_batch(df["CleanComment"].tolist())
    df["Sent_BERT"], df["Conf_BERT"] = b_lbl, b_c

    # Ensemble + collect rows needing GPT
    ens_lbl, ens_conf, source_flags = [], [], []
    todo_indices = []
    for i, (vl, vc, bl, bc) in enumerate(zip(
        df["Sent_VADER"], df["Conf_VADER"],
        df["Sent_BERT"], df["Conf_BERT"]
    )):
        label, conf, src = ensemble_row(vl, vc, bl, bc)
        ens_lbl.append(label)
        ens_conf.append(conf)
        source_flags.append(src)
        if src == "needs_gpt":
            todo_indices.append(i)

    # GPT fallback on unresolved rows
    if todo_indices:
        texts = df.loc[todo_indices, "CleanComment"].tolist()
        g_lbl, g_conf = gpt_refine(texts)
        for idx, row_i in enumerate(todo_indices):
            ens_lbl[row_i]    = g_lbl[idx]
            ens_conf[row_i]   = g_conf[idx]
            source_flags[row_i] = "gpt"

    # Write back to DataFrame
    df["Sentiment_Final"]   = ens_lbl
    df["Conf_Sentiment"]    = ens_conf
    df["Sentiment_Source"]  = source_flags
    return df
