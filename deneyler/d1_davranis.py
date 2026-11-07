# -*- coding: utf-8 -*-
"""
DENEY 1: Davranış yönlerini bul ve yönlendir.

Kullanım:  python -m deneyler.d1_davranis D:/ameliyat_modeller/Qwen2.5-1.5B-Instruct
Çıktı:     deneyler/sonuclar/<model>/<davranis>.json
             - katman başına ayrışma (Cohen's d) eğrisi
             - seçilen katman, split-half kararlılığı
             - TEST sorularına, talimat VERMEDEN, farklı güçlerde üretilen cevaplar
"""
import os
import sys
import time

import torch

from deneyler import benchmark, motor
from deneyler.davranislar import DAVRANISLAR, EGITIM_SORULARI, TEST_SORULARI

GUCLER = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0]


def calistir(yol, secili=None):
    ad = os.path.basename(yol.rstrip("/\\"))
    model, tok = motor.yukle(yol)
    L = len(motor.katmanlar(model))
    print(f"{ad}: {L} katman, cihaz={motor.CIHAZ}", flush=True)

    for davranis, (ta, tb, aciklama, olcum) in DAVRANISLAR.items():
        if secili and davranis not in secili:
            continue
        if os.path.exists(f"deneyler/sonuclar/{ad}/{davranis}.json"):  # bitmişse atla (kaldığı yerden devam)
            print(f"  {davranis}: zaten var, atlandı", flush=True)
            continue
        t0 = time.time()
        akt_a = motor.aktivasyonlar(model, tok, [motor.bicim(tok, s, ta) for s in EGITIM_SORULARI])
        akt_b = motor.aktivasyonlar(model, tok, [motor.bicim(tok, s, tb) for s in EGITIM_SORULARI])
        fark, d = motor.yon_bul(akt_a, akt_b)
        k = motor.katman_sec(d)
        sh = motor.split_half(akt_a, akt_b, k)
        vektor = fark[k]

        uretimler = {}
        for g in GUCLER:
            with motor.yonlendir(model, vektor, k, g):
                uretimler[str(g)] = [motor.uret(model, tok, s) for s in TEST_SORULARI]

        ek = {}
        if davranis == "dalkavukluk":  # standart ölçüm: Anthropic sycophancy seti
            syco = benchmark.sycophancy()
            for g in GUCLER:
                with motor.yonlendir(model, vektor, k, g):
                    ek[str(g)] = round(benchmark.dalkavukluk_orani(model, tok, syco), 3)

        sonuc = {"sycophancy_benchmark": ek or None,
            "model": ad, "davranis": davranis, "aciklama": aciklama, "olcum": olcum,
            "katman_sayisi": L, "katman": k, "derinlik": round(k / (L - 1), 3),
            "cohen_d": [round(float(x), 3) for x in d], "split_half": round(sh, 3),
            "vektor_norm": round(float(vektor.norm()), 2), "gucler": GUCLER,
            "test_sorulari": TEST_SORULARI, "uretimler": uretimler,
            "sure_sn": round(time.time() - t0, 1),
        }
        motor.kaydet(sonuc, f"deneyler/sonuclar/{ad}/{davranis}.json")
        torch.save(vektor, f"deneyler/sonuclar/{ad}/{davranis}.pt")
        print(f"  {davranis:12s} K{k}/{L} d={float(d[k]):.2f} split-half={sh:.2f} "
              f"({sonuc['sure_sn']} sn)", flush=True)
        print(f"     -4: {uretimler['-4.0'][0][:90]!r}\n     +4: {uretimler['4.0'][0][:90]!r}", flush=True)


if __name__ == "__main__":
    calistir(sys.argv[1], sys.argv[2:] or None)
