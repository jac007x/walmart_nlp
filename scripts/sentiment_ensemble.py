import torch, pandas as pd, numpy as np, yaml, json, asyncio, re
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from openai import OpenAI

CFG = yaml.safe_load(open("config.yaml"))
device = 0 if torch.cuda.is_available() else -1
vader = SentimentIntensityAnalyzer()
bert = pipeline("sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=device,
                batch_size=CFG["batch_sizes"]["transformer"])

# --- Sentiment helpers ---------------------------------------------------- #
def vader_score(txt):
    c = vader.polarity_scores(txt)["compound"]
    if c >= 0.6:  return "strongly positive", c
    if c >= 0.2:  return "positive", c
    if c > -0.2:  return "neutral", c
    if c > -0.6:  return "negative", c
    return "strongly negative", abs(c)

def bert_batch(texts):
    out = bert(texts)
    lbl   = ["positive" if r["label"]=="POSITIVE" else "negative" for r in out]
    score = [r["score"] for r in out]
    return lbl, score

client = OpenAI()

async def gpt_refine(texts):
    msgs = [{"role":"system","content":"Classify sentiment (strongly positive / positive / neutral / negative / strongly negative) and confidence 0‑1 as JSON."}]
    msgs += [{"role":"user","content":t} for t in texts]
    resp = await client.chat.completions.acreate(
        model = CFG["gpt"]["model"],
        messages = msgs,
        max_tokens = 30,
    )
    out = [json.loads(c.message.content) for c in resp.choices]
    return [o["sentiment"] for o in out], [o["confidence"] for o in out]

def ensemble_row(v_lbl, v_c, b_lbl, b_c):
    if v_lbl == b_lbl and max(v_c, b_c) >= 0.4:
        return v_lbl, max(v_c, b_c), "rule"
    return None, None, "needs_gpt"

# --- Public API ----------------------------------------------------------- #
def sentiment_dataframe(df):
    # VADER
    vader_res = df["CleanComment"].apply(lambda t: pd.Series(vader_score(t)))
    df["Sent_VADER"], df["Conf_VADER"] = vader_res[0], vader_res[1]

    # BERT
    b_lbl, b_c = bert_batch(df["CleanComment"].tolist())
    df["Sent_BERT"], df["Conf_BERT"] = b_lbl, b_c

    # Ensemble
    ens_lbl, ens_conf, flag = [], [], []
    todo_idx = []
    for i, (vl, vc, bl, bc) in enumerate(zip(df["Sent_VADER"],df["Conf_VADER"],
                                             df["Sent_BERT"], df["Conf_BERT"])):
        e_lbl, e_conf, src = ensemble_row(vl, vc, bl, bc)
        ens_lbl.append(e_lbl)
        ens_conf.append(e_conf)
        flag.append(src)
        if src == "needs_gpt":
            todo_idx.append(i)

    # GPT only on unresolved rows
    if todo_idx:
        batch = df.loc[todo_idx, "CleanComment"].tolist()
        g_lbl, g_conf = asyncio.run(gpt_refine(batch))
        for j, i in enumerate(todo_idx):
            ens_lbl[i]   = g_lbl[j]
            ens_conf[i]  = g_conf[j]
            flag[i] = "gpt"

    df["Sentiment_Final"]   = ens_lbl
    df["Conf_Sentiment"]    = ens_conf
    df["Sentiment_Source"]  = flag
    return df
