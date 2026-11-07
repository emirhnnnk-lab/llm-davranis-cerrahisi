# -*- coding: utf-8 -*-
"""
DENEY 5: Model Türkçe soruda ara katmanlarda İngilizce mi "düşünüyor"?
(Wendler vd. 2024, "Do Llamas Work in English?" deneyinin Türkçe tekrarı)

Yöntem: LOGIT LENS. Her katmanın son-token aktivasyonu modelin son normu + çıktı katmanından
geçirilir; "bu katmanda model hangi kelimeyi söylemeye en yakın?" ölçülür.
Görev: cevabı tek kelime olan Türkçe boşluk doldurma. Girdi de beklenen çıktı da Türkçe.
Her katmanda iki olasılık izlenir:
  P(Türkçe cevabın ilk token'ı)   ör. "mavi"
  P(İngilizce karşılığın ilk token'ı)  ör. "blue"
Hipotez: ara katmanlarda İngilizce olasılığı Türkçeden yüksek, son katmanlarda Türkçe öne geçer.

Kullanım: python -m deneyler.d5_dil_mercegi D:/ameliyat_modeller/Qwen2.5-1.5B-Instruct
"""
import os
import sys

import torch

from deneyler import motor

# (Türkçe boşluk doldurma istemi, Türkçe cevap, İngilizce karşılık)
ORNEKLER = [
    ('Gökyüzünün rengi "', "mavi", "blue"),
    ('Kar tanesinin rengi "', "beyaz", "white"),
    ('Kediler "miyav", köpekler ise "', "hav", "woof"),
    ('Bir haftada yedi "', "gün", "day"),
    ('Su yüz derecede "', "kayn", "boil"),
    ('Arılar çiçeklerden bal "', "yap", "make"),
    ('Güneş doğudan "', "doğ", "rise"),
    ('Kışın hava çok "', "soğuk", "cold"),
    ('Yazın hava çok "', "sıcak", "hot"),
    ('Ağaçların yaprakları genelde "', "yeşil", "green"),
    ('Kan rengi "', "kırmızı", "red"),
    ('İnsanın iki gözü, bir "', "burun", "nose"),
    ('Kitap okumak için "', "göz", "eye"),
    ('Balıklar suda "', "yüz", "swim"),
    ('Kuşlar gökyüzünde "', "uç", "fly"),
    ('Gece gökyüzünde ay ve "', "yıldız", "star"),
    ('Limonun tadı "', "ekşi", "sour"),
    ('Şekerin tadı "', "tatlı", "sweet"),
    ('Bir yılda on iki "', "ay", "month"),
    ('Ateş çok "', "sıcak", "hot"),
]


def ilk_token(tok, kelime, bosluklu):
    return tok.encode((" " if bosluklu else "") + kelime, add_special_tokens=False)[0]


def calistir(yol):
    ad = os.path.basename(yol.rstrip("/\\"))
    model, tok = motor.yukle(yol)
    norm, bas = model.model.norm, model.lm_head
    L = len(motor.katmanlar(model))
    tr_egri, en_egri, detay = torch.zeros(L), torch.zeros(L), []
    for istem, tr, en in ORNEKLER:
        # v2: few-shot. v1'de tek başına istem verildi ve model son katmanda bile Türkçe cevabı
        # tahmin etmedi (P~0.0003) -> görev çalışmadı, ara katman yorumu anlamsızdı.
        ornekler = [o for o in ORNEKLER if o[0] != istem][:4]
        onek = "".join(f'{i}{t}"\n' for i, t, _ in ornekler)
        ids = motor._ids(tok, onek + istem)  # sohbet şablonu yok: ham metin tamamlama
        hs = model(ids, output_hidden_states=True).hidden_states[1:]
        t_id, e_id = ilk_token(tok, tr, False), ilk_token(tok, en, False)
        satir = []
        for k, h in enumerate(hs):
            p = torch.softmax(bas(norm(h[0, -1])).float(), -1)
            tr_egri[k] += float(p[t_id])
            en_egri[k] += float(p[e_id])
            satir.append({"tr": round(float(p[t_id]), 4), "en": round(float(p[e_id]), 4),
                          "en_iyi": tok.decode([int(p.argmax())])})
        detay.append({"istem": istem, "tr": tr, "en": en, "katmanlar": satir})
    n = len(ORNEKLER)
    sonuc = {"model": ad, "katman_sayisi": L, "n": n,
             "p_turkce": [round(float(x / n), 4) for x in tr_egri],
             "p_ingilizce": [round(float(x / n), 4) for x in en_egri],
             "detay": detay}
    ust = [k for k in range(L) if sonuc["p_ingilizce"][k] > sonuc["p_turkce"][k]]
    sonuc["ingilizce_onde_katmanlar"] = ust
    sonuc["surum"] = "v2-few-shot"
    sonuc["son_katman_dogru"] = round(sum(d["katmanlar"][-1]["en_iyi"].strip().lower() and
                                          tr.startswith(d["katmanlar"][-1]["en_iyi"].strip().lower())
                                          for d, (_, tr, _) in zip(detay, ORNEKLER)) / n, 3)
    motor.kaydet(sonuc, f"deneyler/sonuclar/{ad}/dil_mercegi.json")
    print(f"{ad}: son katmanda doğru Türkçe cevap oranı {sonuc['son_katman_dogru']} | İngilizce önde: {ust}", flush=True)
    for k in range(0, L, max(1, L // 8)):
        print(f"  K{k:2d}  TR={sonuc['p_turkce'][k]:.4f}  EN={sonuc['p_ingilizce'][k]:.4f}", flush=True)


if __name__ == "__main__":
    calistir(sys.argv[1])
