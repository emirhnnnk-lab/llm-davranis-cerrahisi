# -*- coding: utf-8 -*-
"""sonuclar/*.json -> grafikler/*.png"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KOK = os.path.dirname(os.path.abspath(__file__))
S, G = os.path.join(KOK, "sonuclar"), os.path.join(KOK, "grafikler")
MODELLER = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", "SmolLM2-1.7B-Instruct"]
DAV = ["dalkavukluk", "uzunluk", "ozguven", "mutluluk", "ofke", "resmiyet", "turkce"]
RENK = ["#2a9d8f", "#e76f51", "#264653"]


def oku(m, ad):
    y = os.path.join(S, m, ad + ".json")
    return json.load(open(y, encoding="utf-8")) if os.path.exists(y) else None


# 1) Davranış yönlerinin göreli derinliği
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, m in enumerate(MODELLER):
    xs, ys = [], []
    for j, d in enumerate(DAV):
        r = oku(m, d)
        if r:
            xs.append(r["katman"] / (r["katman_sayisi"] - 1)); ys.append(j + (i - 1) * 0.18)
    s = oku(m, "sansur")
    if s and s["split_half"] > 0.3:
        xs.append(s["derinlik"]); ys.append(len(DAV) + (i - 1) * 0.18)
    ax.scatter(xs, ys, s=60, color=RENK[i], label=m.replace("-Instruct", ""))
ax.set_yticks(range(len(DAV) + 1)); ax.set_yticklabels(DAV + ["reddetme"])
ax.set_xlabel("Göreli katman derinliği (0 = giriş, 1 = çıkış)"); ax.set_xlim(0, 1)
ax.set_title("Davranışlar modelin hangi derinliğinde kodlanıyor?")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(G, "1_davranis_derinligi.png"), dpi=130)

# 2) Doz-tepki (hakem puanı) — Qwen 1.5B
m = "Qwen2.5-1.5B-Instruct"
fig, ax = plt.subplots(figsize=(7, 4))
for d in ["mutluluk", "ofke", "resmiyet", "ozguven", "dalkavukluk"]:
    r = oku(m, d)
    if r and r.get("olcum_sonuc"):
        o = r["olcum_sonuc"]
        g = sorted(o, key=float); ax.plot([float(x) for x in g], [o[x]["ortalama"] for x in g], marker="o", label=d)
ax.set_xlabel("Yönlendirme gücü"); ax.set_ylabel("Hakem puanı (1-5)")
ax.set_title(f"Doz-tepki: {m}"); ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(os.path.join(G, "2_doz_tepki_15b.png"), dpi=130)

# 3) 14B: orijinal vs topluluk sansürsüz
k = json.load(open(os.path.join(S, "buyuk_model_14b.json"), encoding="utf-8"))
etk = ["red_harmbench", "red_xstest", "mmlu", "gsm8k"]
ad = ["Zararlıya red", "Zararsıza red\n(aşırı red)", "MMLU", "GSM8K"]
fig, ax = plt.subplots(figsize=(7, 4)); w = .38
for i, (isim, v) in enumerate(k.items()):
    ax.bar([x + (i - .5) * w for x in range(4)], [v[e] for e in etk], w, color=RENK[i], label=v["etiket"])
ax.set_xticks(range(4)); ax.set_xticklabels(ad); ax.set_ylim(0, 1)
ax.set_title("Qwen2.5 14B: reddetme kaldırılınca yetenek kaybı"); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(G, "3_14b_karsilastirma.png"), dpi=130)

# 4) GSM8K artefaktı
o = json.load(open(os.path.join(S, m, "gsm_kontrol_ozet.json"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(6, 3.8))
for i, et in enumerate(["orijinal", "kaldirma"]):
    v = [o[f"{et}_160"]["dogru"], o[f"{et}_512"]["dogru"]]
    ax.bar([x + (i - .5) * w for x in range(2)], v, w, color=RENK[i], label="orijinal" if i == 0 else "reddetme kaldırılmış")
ax.set_xticks(range(2)); ax.set_xticklabels(["160 token sınırı\n(orijinalde 19/20 kesik)", "512 token sınırı"])
ax.set_ylabel("Doğru (20 soruda)"); ax.set_title("GSM8K 'artışı' bir ölçüm artefaktıydı"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(G, "4_gsm_artefakt.png"), dpi=130)
print("tamam")
