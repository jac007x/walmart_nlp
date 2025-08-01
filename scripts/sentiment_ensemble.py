#!/usr/bin/env python
"""
Sentiment ensemble for Walmart BU comment pipeline.

Stages:
  1) VADER (fast, rule-based)
  2) DistilBERT via HuggingFace pipeline (batch; truncates >512 tokens)
  3) GPT-4o-mini fallback for any unresolved cases

Produces final sentiment label + confidence.
"""
import torch
import pandas as pd
import numpy as np
import yaml
import json
import re

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

# --- Config & Model Initialization -------------------------------------- #
CFG = yaml.safe_load(open("config.yaml"))

device = 0 if torch.cuda.is_available() else -1

# 1) VADER
vader = SentimentIntensityAnalyzer()

# 2) DistilBERT sentiment pipeline
bert = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    batch_size=CFG["batch_sizes"]["transformer"],
    tokenizer_kwargs={
        "truncation": True,
        "max_length": 512
    }
)

# 3) OpenAI GPT client
client = OpenAI()

# --- Helper Functions ---------------------------------------------------- #
def vader_score(txt: str) -> tuple[str, float]:
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

def bert_batch(texts: list[str]) -> tuple[list[str], list[float]]:
    out   = bert(texts)
    lbl   = ["positive" if r["label"] == "POSITIVE" else "negative" for r in out]
    score = [r["score"] for r in out]
    return lbl, score

def gpt_refine(texts: list[str]) -> tuple[list[str], list[float]]:
    """
    One completion per text to preserve ordering.
    Returns two lists: [sentiment, ...], [confidence, ...].
    """
    sentiments = []
    confidences = []
    for text in texts:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify sentiment as one of: strongly positive, positive, neutral, "
                    "negative, strongly negative. Then return JSON with keys "
                    '"sentiment" and "confidence" (0.0–1.0).'
                )
            },
            {"role": "user", "content": text}
        ]
        resp = client.chat.completions.create(
            model      = CFG["gpt"]["model"],
            messages   = messages,
            max_tokens = 30,
        )
        parsed = json.loads(resp.choices[0].message.content)
        sentiments.append(parsed["sentiment"])
        confidences.append(parsed["confidence"])
    return sentiments, confidences

def ensemble_row(
    v_lbl: str, v_conf: float,
    b_lbl: str, b_conf: float
) -> tuple[str|None, float|None, str]:
    """
    If VADER & BERT agree with confidence ≥0.4, take that.
    Otherwise mark for GPT.
    """
    if v_lbl == b_lbl and max(v_conf, b_conf) >= 0.4:
        return v_lbl, max(v_conf, b_conf), "rule"
    return None, None, "needs_gpt"

# --- Public API ---------------------------------------------------------- #
def sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # VADER pass
    vader_res = df["CleanComment"].apply(lambda t: pd.Series(vader_score(t)))
    df["Sent_VADER"], df["Conf_VADER"] = vader_res[0], vader_res[1]

    # BERT pass
    b_lbl, b_conf = bert_batch(df["CleanComment"].tolist())
    df["Sent_BERT"], df["Conf_BERT"] = b_lbl, b_conf

    # Ensemble + collect rows for GPT
    ens_lbl = []
    ens_conf = []
    source  = []
    to_gpt   = []

    for i, (vl, vc, bl, bc) in enumerate(zip(
        df["Sent_VADER"], df["Conf_VADER"],
        df["Sent_BERT"], df["Conf_BERT"]
    )):
        lbl, conf, src = ensemble_row(vl, vc, bl, bc)
        ens_lbl.append(lbl)
        ens_conf.append(conf)
        source.append(src)
        if src == "needs_gpt":
            to_gpt.append(i)

    # GPT fallback
    if to_gpt:
        texts = df.loc[to_gpt, "CleanComment"].tolist()
        g_lbls, g_confs = gpt_refine(texts)
        for idx, row_i in enumerate(to_gpt):
            ens_lbl[row_i]  = g_lbls[idx]
            ens_conf[row_i] = g_confs[idx]
            source[row_i]   = "gpt"

    df["Sentiment_Final"]  = ens_lbl
    df["Conf_Sentiment"]   = ens_conf
    df["Sentiment_Source"] = source
    return df
