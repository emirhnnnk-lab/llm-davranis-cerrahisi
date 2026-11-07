# -*- coding: utf-8 -*-
"""
Genel davranış cerrahisi motoru: herhangi bir HF modelinde (Qwen, Llama, SmolLM...)
1) zıt iki talimat altında aktivasyon topla,
2) her katmanda davranış yönünü bul (ortalama farkı),
3) yönün en net olduğu katmanı ÖLÇEREK seç (Cohen's d),
4) yönü tek katmanda ekleyerek/çıkararak davranışı değiştir (steering),
5) yönü tüm katmanlarda silerek davranışı kaldır (ablasyon).

Panelin cekirdek/ modülündeki 1.5B'ye özel kodun genelleştirilmiş hâli.
"""
import contextlib
import json
import os

import torch

torch.set_grad_enabled(False)
CIHAZ = "cuda" if torch.cuda.is_available() else "cpu"


def yukle(yol):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(yol)
    dtype = torch.float16 if CIHAZ == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(yol, dtype=dtype).to(CIHAZ)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(yol, torch_dtype=dtype).to(CIHAZ)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def katmanlar(model):
    return model.model.layers


def bicim(tok, soru, talimat=None):
    """Talimat varsa kullanıcı mesajının başına eklenir (her model system rolünü desteklemez)."""
    icerik = f"{talimat}\n\n{soru}" if talimat else soru
    return tok.apply_chat_template([{"role": "user", "content": icerik}],
                                   add_generation_prompt=True, tokenize=False)


def _ids(tok, metin):
    return tok(metin, return_tensors="pt", add_special_tokens=False).input_ids.to(CIHAZ)


def aktivasyonlar(model, tok, metinler):
    """Her metnin SON token'ında tüm katmanların çıktısı. Döner: (n, L, d) float32, CPU."""
    yigin = []
    for m in metinler:
        cikti = model(_ids(tok, m), output_hidden_states=True)
        hs = torch.stack([h[0, -1, :] for h in cikti.hidden_states[1:]])  # embedding hariç
        yigin.append(hs.float().cpu())
    return torch.stack(yigin)


def yon_bul(akt_a, akt_b):
    """
    Her katman için: ham fark vektörü (A-B ortalaması) ve ayrışma (Cohen's d).
    d: iki grubun yön üzerindeki izdüşümleri birbirinden kaç standart sapma uzak.
    """
    fark = akt_a.mean(0) - akt_b.mean(0)                     # (L, d)
    birim = fark / fark.norm(dim=-1, keepdim=True)
    pa = (akt_a * birim).sum(-1)                              # (n, L)
    pb = (akt_b * birim).sum(-1)
    sd = ((pa.var(0) + pb.var(0)) / 2).sqrt()
    d = (pa.mean(0) - pb.mean(0)) / sd
    return fark, d


def katman_sec(d, alt=0.2, ust=0.8):
    """Ayrışmanın en yüksek olduğu katman; çok erken/çok geç katmanlar steering için kötü."""
    L = len(d)
    a, b = int(L * alt), max(int(L * ust), int(L * alt) + 1)
    return a + int(torch.argmax(d[a:b]))


def split_half(akt_a, akt_b, katman):
    """Yön verinin iki yarısından ayrı ayrı bulununca aynı mı? (kararlılık, kosinüs)"""
    n = len(akt_a) // 2
    v1 = akt_a[:n, katman].mean(0) - akt_b[:n, katman].mean(0)
    v2 = akt_a[n:, katman].mean(0) - akt_b[n:, katman].mean(0)
    return float(torch.nn.functional.cosine_similarity(v1, v2, dim=0))


def _uygula(cikti, f):
    if isinstance(cikti, tuple):
        return (f(cikti[0]),) + cikti[1:]
    return f(cikti)


@contextlib.contextmanager
def yonlendir(model, vektor, katman, guc):
    """h <- h + guc * vektor   (tek katman, tüm pozisyonlar). guc<0 zıt davranış."""
    v = vektor.to(CIHAZ)
    def kanca(mod, girdi, cikti):
        return _uygula(cikti, lambda h: (h.float() + guc * v).to(h.dtype))
    kol = katmanlar(model)[katman].register_forward_hook(kanca)
    try:
        yield
    finally:
        kol.remove()


@contextlib.contextmanager
def ablasyon(model, vektor, alpha=1.0):
    """h <- h - alpha (h·v̂) v̂   tüm katmanlarda: yön modelin hiçbir yerinde kalmaz."""
    v = (vektor / vektor.norm()).to(CIHAZ).float()
    def kanca(mod, girdi, cikti):
        def f(h):
            h32 = h.float()
            return (h32 - alpha * (h32 @ v).unsqueeze(-1) * v).to(h.dtype)
        return _uygula(cikti, f)
    kollar = [k.register_forward_hook(kanca) for k in katmanlar(model)]
    try:
        yield
    finally:
        for k in kollar:
            k.remove()


@contextlib.contextmanager
def kosullu_ekle(model, kosul_v, kosul_katman, esik, eylem_v, eylem_katman, guc):
    """
    KOŞULLU YÖNLENDİRME (sansür EKLEME; Lee vd. 2024, "Conditional Activation Steering" fikri):
    Prompt'un kosul_katman'daki aktivasyonu konu yönüne (kosul_v) eşikten fazla benziyorsa,
    eylem yönü (reddetme) eylem_katman'a eklenir. Konu dışındaki sorulara dokunulmaz.
    """
    kv = (kosul_v / kosul_v.norm()).to(CIHAZ).float()
    ev = eylem_v.to(CIHAZ)
    durum = {"aktif": None, "benzerlik": None}

    def kosul_kanca(mod, girdi, cikti):
        h = cikti[0] if isinstance(cikti, tuple) else cikti
        if durum["aktif"] is None:  # sadece prompt'un ilk geçişinde karar ver
            son = h[0, -1].float()
            durum["benzerlik"] = float(son @ kv / son.norm())
            durum["aktif"] = durum["benzerlik"] > esik
        return cikti

    def eylem_kanca(mod, girdi, cikti):
        if not durum["aktif"]:
            return cikti
        return _uygula(cikti, lambda h: (h.float() + guc * ev).to(h.dtype))

    k1 = katmanlar(model)[kosul_katman].register_forward_hook(kosul_kanca)
    k2 = katmanlar(model)[eylem_katman].register_forward_hook(eylem_kanca)
    try:
        yield durum
    finally:
        k1.remove()
        k2.remove()


def uret(model, tok, soru, max_yeni=50, talimat=None):
    ids = _ids(tok, bicim(tok, soru, talimat))
    cikti = model.generate(ids, max_new_tokens=max_yeni, do_sample=False,
                           pad_token_id=tok.pad_token_id, repetition_penalty=1.1)
    return tok.decode(cikti[0, ids.shape[1]:], skip_special_tokens=True).strip()


def kaydet(veri, yol):
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    with open(yol, "w", encoding="utf-8") as f:
        json.dump(veri, f, ensure_ascii=False, indent=1)
