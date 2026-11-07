# -*- coding: utf-8 -*-
"""
Üretilen cevapları ölçer ve sonuç dosyasına "olcum" alanı ekler.

- uzunluk : kelime sayısı (kesin ölçüm)
- dil     : Türkçe cevap oranı (Türkçe harf/kelime sezgisi, kesin sayılabilir)
- hakem   : ayrı bir LLM (Ollama, varsayılan qwen2.5:3b) her cevaba 1-5 puan verir.
            Hakem steering yapılan modelden FARKLI bir modeldir; kendi kendini puanlama yok.
- bozulma : cevap tekrar döngüsüne girdi mi / boş mu (yönlendirme modeli bozdu mu?)

Kullanım: python -m deneyler.olc deneyler/sonuclar/Qwen2.5-1.5B-Instruct
"""
import glob
import json
import os
import re
import sys

import requests

from deneyler.davranislar import HAKEM_SORUSU

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
HAKEM = os.environ.get("HAKEM_MODEL", "qwen2.5:3b")
_S = requests.Session()
_S.trust_env = False

TR_KELIMELER = {"ve", "bir", "bu", "için", "çok", "ile", "da", "de", "gibi", "ama", "olarak", "ben", "sen"}


def turkce_mi(metin):
    harf = len(re.findall(r"[çğışöüÇĞİŞÖÜ]", metin))
    kelime = sum(w in TR_KELIMELER for w in re.findall(r"\w+", metin.lower()))
    return harf + 2 * kelime >= 3


def bozuk_mu(metin):
    kel = metin.split()
    if len(kel) < 3:
        return True
    # aynı 3'lü kelime grubu 3+ kez tekrarlanıyorsa döngü
    uclu = [" ".join(kel[i:i + 3]) for i in range(len(kel) - 2)]
    return max(uclu.count(u) for u in set(uclu)) >= 3


def hakem_puani(soru, metin):
    istem = (f"{soru}\nAnswer with a single digit from 1 to 5 and nothing else.\n\nTEXT:\n{metin[:600]}")
    r = _S.post(f"{OLLAMA}/api/generate", timeout=120, json={
        "model": HAKEM, "prompt": istem, "stream": False,
        "options": {"temperature": 0, "num_predict": 3}})
    m = re.search(r"[1-5]", r.json().get("response", ""))
    return int(m.group()) if m else None


def olc(dosya):
    with open(dosya, encoding="utf-8") as f:
        s = json.load(f)
    olcum = {}
    for guc, cevaplar in s["uretimler"].items():
        if s["olcum"] == "uzunluk":
            degerler = [len(c.split()) for c in cevaplar]
        elif s["olcum"] == "dil":
            degerler = [1.0 if turkce_mi(c) else 0.0 for c in cevaplar]
        else:
            degerler = [hakem_puani(HAKEM_SORUSU[s["davranis"]], c) for c in cevaplar]
        gecerli = [x for x in degerler if x is not None]
        olcum[guc] = {
            "ortalama": round(sum(gecerli) / len(gecerli), 3) if gecerli else None,
            "degerler": degerler,
            "bozulma_orani": round(sum(bozuk_mu(c) for c in cevaplar) / len(cevaplar), 3),
        }
    s["olcum_sonuc"] = olcum
    s["hakem"] = HAKEM if s["olcum"] == "hakem" else None
    with open(dosya, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    satir = "  ".join(f"{g}:{v['ortalama']}" for g, v in olcum.items())
    print(f"{s['model']:28s} {s['davranis']:12s} {satir}", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        for dosya in sorted(glob.glob(os.path.join(d, "*.json"))):
            if "uretimler" in open(dosya, encoding="utf-8").read(200000):
                olc(dosya)
