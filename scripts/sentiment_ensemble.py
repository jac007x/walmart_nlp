import torch
import pandas as pd
import numpy as np
import yaml
import json
import re

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

# Load config
CFG = yaml.safe_load(open("config.yaml"))

# Initialize models
device = 0 if torch.cuda.is_available() else -1
vader = SentimentIntensityAnalyzer()
bert  = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    batch_size=CFG["batch_sizes"]["transformer"],
)
client = OpenAI()

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

def bert_batch(texts: list[str]):
    out   = bert(texts)
    lbl   = ["positive" if r["label"]=="POSITIVE" else "negative" for r in out]
    score = [r["score"] for r in out]
    return lbl, score

def gpt_refine(texts: list[str]) -> tuple[list[str], list[float]]:
    """
    Sends each text as its own completion to ensure 1:1 feedback.
    Returns two lists: [sentiment,...] and [confidence,...].
    """
    results = []
    for text in texts:
        messages = [
            {
                "role":"system",
                "content":(
                    "Classify sentiment as one of: strongly positive, positive, "
                    "neutral, negative, strongly negative. Then return JSON "
                    'with keys "sentiment" and "confidence" (0.0–1.0).'
                )
            },
            {"role":"user","content": text}
        ]
        resp = client.chat.completions.create(
            model      = CFG["gpt"]["model"],
            messages   = messages,
            max_tokens = 30,
        )
        parsed = json.loads(resp.choices[0].message.content)
        results.append(parsed)
    sentiments  = [r["sentiment"]  for r in results]
    confidences = [r["confidence"] for r in results]
    return sentiments, confidences

def ensemble_row(v_lbl, v_c, b_lbl, b_c):
    if v_lbl == b_lbl and max(v_c, b_c) >= 0.4:
        return v_lbl, max(v_c, b_c), "rule"
    return None, None, "needs_gpt"

def sentiment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # VADER
    vader_res = df["CleanComment"].apply(lambda t: pd.Series(vader_score(t)))
    df["Sent_VADER"], df["Conf_VADER"] = vader_res[0], vader_res[1]

    # BERT
    b_lbl, b_c = bert_batch(df["CleanComment"].tolist())
    df["Sent_BERT"], df["Conf_BERT"] = b_lbl, b_c

    # Ensemble + collect which rows need GPT
    ens_lbl, ens_conf, src_flag = [], [], []
    needs_gpt = []
    for i, (vl, vc, bl, bc) in enumerate(zip(
        df["Sent_VADER"], df["Conf_VADER"],
        df["Sent_BERT"], df["Conf_BERT"]
    )):
        lbl, conf, src = ensemble_row(vl, vc, bl, bc)
        ens_lbl.append(lbl)
        ens_conf.append(conf)
        src_flag.append(src)
        if src == "needs_gpt":
            needs_gpt.append(i)

    # GPT fallback, one at a time
    if needs_gpt:
        texts = df.loc[needs_gpt, "CleanComment"].tolist()
        g_lbls, g_confs = gpt_refine(texts)
        for idx, row_i in enumerate(needs_gpt):
            ens_lbl[row_i]   = g_lbls[idx]
            ens_conf[row_i]  = g_confs[idx]
            src_flag[row_i]  = "gpt"

    df["Sentiment_Final"]  = ens_lbl
    df["Conf_Sentiment"]   = ens_conf
    df["Sentiment_Source"] = src_flag
    return df
