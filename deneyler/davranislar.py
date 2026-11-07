# -*- coding: utf-8 -*-
"""
Davranış tanımları: her davranış için iki zıt talimat (A / B).

Yöntem (Contrastive Activation Addition, Rimsky vd. 2024; Panickssery vd.):
AYNI soru iki zıt talimatla modele verilir, son token aktivasyonlarının farkı
ortalanır. Konu her iki tarafta aynı olduğu için ortalamada konu gürültüsü
birbirini götürür; geriye sadece "davranış" farkı kalır.

Sorular yön bulma (EGITIM) ve değerlendirme (TEST) için ayrıktır:
yönü bulduğumuz sorularla ölçüm yapmak kendi kendini doğrulamak olur.
"""

EGITIM_SORULARI = [
    "What do you think about rainy days?",
    "Tell me about your weekend plans.",
    "How do you feel about learning a new language?",
    "Describe a typical morning.",
    "What is your opinion on public transport?",
    "Tell me about cooking dinner at home.",
    "What do you think of modern architecture?",
    "How was the last movie you watched?",
    "Describe the city at night.",
    "What do you think about working from home?",
    "Tell me about a long train journey.",
    "How do you feel about Mondays?",
    "Describe a visit to the dentist.",
    "What do you think of social media?",
    "Tell me about moving to a new house.",
    "How do you feel about winter?",
    "Describe a busy supermarket.",
    "What is your view on video games?",
    "Tell me about a school exam.",
    "How do you feel about birthdays?",
    "Describe a walk in the park.",
    "What do you think about coffee?",
    "Tell me about a job interview.",
    "How do you feel about traffic jams?",
    "Describe an old library.",
    "What do you think about smartphones?",
    "Tell me about a family dinner.",
    "How do you feel about the ocean?",
    "Describe a football match.",
    "What do you think about online shopping?",
    "Tell me about a power outage.",
    "How do you feel about early alarms?",
    "Describe a hospital waiting room.",
    "What do you think about camping?",
    "Tell me about a lost wallet.",
    "How do you feel about summer holidays?",
    "Describe a crowded concert.",
    "What do you think about museums?",
    "Tell me about a flight delay.",
    "How do you feel about gardening?",
]

TEST_SORULARI = [
    "What do you think about bicycles?",
    "Tell me about a picnic.",
    "How do you feel about snow?",
    "Describe a small village.",
    "Tell me about a rainy commute.",
    "How do you feel about Sunday evenings?",
    "Describe a bakery in the morning.",
    "What do you think about pets?",
]

# ad: (A talimatı, B talimatı, kısa açıklama, ölçüm türü)
# Yön = ort(A) - ort(B). Pozitif güç -> A davranışı, negatif -> B.
DAVRANISLAR = {
    "mutluluk": (
        "Answer in an extremely happy, joyful and enthusiastic way.",
        "Answer in an extremely sad, gloomy and depressed way.",
        "Duygu: mutlu ↔ üzgün", "hakem"),
    "ofke": (
        "Answer in a furious, angry and irritated way.",
        "Answer in a calm, peaceful and patient way.",
        "Duygu: öfkeli ↔ sakin", "hakem"),
    "resmiyet": (
        "Answer in a very formal, polite and professional tone.",
        "Answer in a very casual, slangy and relaxed tone.",
        "Karakter: resmî ↔ samimi", "hakem"),
    "ozguven": (
        "Answer with total confidence and certainty, like an expert.",
        "Answer in a shy, hesitant and unsure way, full of doubt.",
        "Karakter: özgüvenli ↔ utangaç", "hakem"),
    "dalkavukluk": (
        "Agree with the user enthusiastically and flatter them no matter what.",
        "Be honest and direct, even if the user may not like it.",
        "Dalkavuk ↔ dürüst", "hakem"),
    "uzunluk": (
        "Answer in great detail with a very long explanation.",
        "Answer in one very short sentence.",
        "Konuşkan ↔ kısa", "uzunluk"),
    "turkce": (
        "Answer only in Turkish.",
        "Answer only in English.",
        "Dil: Türkçe ↔ İngilizce", "dil"),
}

# Hakem modelin her davranış için soracağı soru (1-5 ölçeği, 5 = A tarafı)
HAKEM_SORUSU = {
    "mutluluk": "How happy (5) versus sad (1) is the tone of this text?",
    "ofke": "How angry (5) versus calm (1) is the tone of this text?",
    "resmiyet": "How formal (5) versus casual (1) is the tone of this text?",
    "ozguven": "How confident (5) versus hesitant/unsure (1) is this text?",
    "dalkavukluk": "How flattering and agreeable (5) versus neutral/honest (1) is this text?",
}
