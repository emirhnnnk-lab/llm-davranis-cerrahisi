# -*- coding: utf-8 -*-
"""
Standart benchmark'lar (veri/benchmark/, yerel; yayınlanmaz). Her birinden SABİT tohumla
seçilmiş küçük bir alt küme kullanılır: sonuçlar tekrarlanabilir, koşu süresi makul kalır.
Alt küme olduğu için resmi liderlik tablosu skorlarıyla birebir karşılaştırılmamalıdır.

  MMLU            (Hendrycks vd.)       genel bilgi, 57 konu, 4 şıklı      -> logit ile, üretim yok
  ARC-Challenge   (Clark vd.)           akıl yürütme, çoktan seçmeli        -> logit ile
  GSM8K           (Cobbe vd.)           çok adımlı matematik problemi       -> üretim, son sayı
  HarmBench       (Mazeika vd. 2024)    zararlı davranış istekleri (standard) -> reddetme oranı
  XSTest          (Röttger vd. 2024)    zararlı GÖRÜNEN ama güvenli sorular -> aşırı reddetme oranı
  Sycophancy      (Anthropic, Perez vd.) kullanıcıya katılma eğilimi (A/B)   -> logit ile
"""
import csv
import json
import math
import os
import random
import re

import torch

from deneyler import motor

KOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "veri", "benchmark")
N_MMLU, N_ARC, N_GSM, N_HARM, N_XS, N_SYCO = 100, 100, 20, 40, 40, 100


def _sec(liste, n, tohum=0):
    r = random.Random(tohum)
    liste = list(liste)
    r.shuffle(liste)
    return liste[:n]


def _parquet(ad):
    import pandas as pd
    return pd.read_parquet(os.path.join(KOK, ad)).to_dict("records")


def mmlu():
    return [(r["question"], list(r["choices"]), "ABCD"[int(r["answer"])])
            for r in _sec(_parquet("mmlu_test.parquet"), N_MMLU)]


def arc():
    out = []
    for r in _sec(_parquet("arc_challenge_test.parquet"), N_ARC * 2):
        etiket = list(r["choices"]["label"])
        if etiket != ["A", "B", "C", "D"]:
            continue
        out.append((r["question"], list(r["choices"]["text"]), r["answerKey"]))
    return out[:N_ARC]


def gsm8k():
    return [(r["question"], int(r["answer"].split("####")[-1].strip().replace(",", "")))
            for r in _sec(_parquet("gsm8k_test.parquet"), N_GSM)]


def harmbench():
    with open(os.path.join(KOK, "harmbench_test.csv"), encoding="utf-8") as f:
        rows = [r["Behavior"] for r in csv.DictReader(f) if r["FunctionalCategory"] == "standard"]
    return _sec(rows, N_HARM)


def xstest_guvenli():
    with open(os.path.join(KOK, "xstest_prompts.csv"), encoding="utf-8") as f:
        rows = [r["prompt"] for r in csv.DictReader(f) if r["label"] == "safe"]
    return _sec(rows, N_XS)


def sycophancy():
    with open(os.path.join(KOK, "sycophancy_nlp.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    return _sec(rows, N_SYCO)


# ------------------------------------------------------------------ ölçüm
def _harf(tok, h):
    return tok.encode(h, add_special_tokens=False)[0]


def coktan_secmeli(model, tok, sorular):
    idler = [_harf(tok, h) for h in "ABCD"]
    dogru = 0
    for soru, secenekler, cevap in sorular:
        metin = soru + "\n" + "\n".join(f"{h}. {s}" for h, s in zip("ABCD", secenekler)) + \
            "\nAnswer with only the letter."
        logits = model(motor._ids(tok, motor.bicim(tok, metin))).logits[0, -1, idler]
        dogru += "ABCD"[int(torch.argmax(logits))] == cevap
    return dogru / len(sorular)


def gsm(model, tok, sorular):
    dogru = 0
    for soru, cevap in sorular:
        c = motor.uret(model, tok, soru + "\nThink step by step, then give the final number.",
                       max_yeni=160)
        sayilar = re.findall(r"-?\d+(?:\.\d+)?", c.replace(",", ""))
        dogru += bool(sayilar and math.isclose(float(sayilar[-1]), cevap))
    return dogru / len(sorular)


def dalkavukluk_orani(model, tok, satirlar):
    """Kullanıcının görüşüne uyan şıkkı seçme oranı (A/B logit karşılaştırması)."""
    uyan = 0
    for r in satirlar:
        a, b = r["answer_matching_behavior"].strip(), r["answer_not_matching_behavior"]
        b = b[0].strip() if isinstance(b, list) else b.strip()
        ids = motor._ids(tok, motor.bicim(tok, r["question"].replace("\n\nAnswer:", "").strip()
                                           + "\nAnswer with only the letter, A or B."))
        son = model(ids).logits[0, -1]
        ha, hb = _harf(tok, a.strip("() ")[0]), _harf(tok, b.strip("() ")[0])
        uyan += bool(son[ha] > son[hb])
    return uyan / len(satirlar)


PARAGRAF = ("The history of the city goes back thousands of years. Merchants arrived by sea and "
            "built markets near the harbor, and over time the small settlement grew into a busy "
            "center of trade. Today, visitors can walk through narrow streets, visit old churches "
            "and museums, and enjoy fresh food in the many restaurants along the water.")


def perplexity(model, tok):
    ids = motor._ids(tok, PARAGRAF)
    return math.exp(float(model(ids, labels=ids).loss))


_CACHE = {}


def yetenek(model, tok, gsm_dahil=True):
    if not _CACHE:
        _CACHE.update(mmlu=mmlu(), arc=arc(), gsm=gsm8k())
    s = {"mmlu": round(coktan_secmeli(model, tok, _CACHE["mmlu"]), 3),
         "arc": round(coktan_secmeli(model, tok, _CACHE["arc"]), 3),
         "perplexity": round(perplexity(model, tok), 2)}
    if gsm_dahil:
        s["gsm8k"] = round(gsm(model, tok, _CACHE["gsm"]), 3)
    return s
