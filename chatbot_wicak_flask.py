import json
import os
import re
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# ==============================================================================
# 1. INISIALISASI FLASK & KONFIGURASI
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base_wicak_pknstan.json"

app = Flask(__name__, static_folder=str(BASE_DIR))
# Mengizinkan panggilan dari index.html lokal maupun GitHub Pages
CORS(app, resources={r"/api/*": {"origins": "*"}})

STOPWORDS_ID = {
    "yang", "di", "ke", "dari", "dan", "atau", "untuk", "pada", "dengan",
    "adalah", "itu", "ini", "ada", "tidak", "bisa", "apa", "bagaimana",
    "kapan", "dimana", "siapa", "mengapa", "kenapa", "berapa", "apakah",
    "boleh", "saya", "aku", "kamu", "anda", "kita", "kami", "jika", "kalau",
    "bila", "agar", "supaya", "sebagai", "oleh", "dalam", "tentang", "serta",
    "juga", "sudah", "belum", "akan", "sedang", "harus", "wajib", "dapat",
    "ya", "dong", "nih", "sih", "kah", "pun", "lah", "nya",
}

SYNONYM_MAP = {
    "do": "drop out putus studi berhenti kuliah",
    "nyontek": "menyontek kecurangan akademik plagiarisme ujian",
    "vape": "rokok elektrik pod merokok kawasan tanpa rokok",
    "cumlaude": "pujian yudisium ipk predikat kelulusan",
    "skbp": "surat keterangan bebas pustaka perpustakaan skripsi ktta",
    "bimbel": "bimbingan belajar try out spmb penipuan resmi",
    "booking": "reservasi peminjaman ruang diskusi sewa gedung",
    "submit": "kirim unggah naskah artikel jurnal ojs",
    "telat": "terlambat denda sanksi pengembalian",
    "bolos": "absen kehadiran alfa presensi kuliah",
    "ukt": "biaya pendidikan gratis ikatan dinas",
    "asrama": "pendidikan karakter kapasitas hunian mahasiswa baru",
}

SYSTEM_PROMPT = """
Kamu adalah **Wicak** (dari kata *Wicaksana*), asisten virtual resmi PKN STAN yang ramah, bijaksana, dan membantu.

Gaya Bahasa Wajib:
1. Natural, hangat, dan manusiawi—bukan seperti membaca template kaku.
2. Langsung ke inti jawaban di kalimat pertama (misalnya: "Boleh.", "Tidak.", "Tidak otomatis bersalah.", atau jawaban ringkas yang tegas), baru diikuti penjelasan seperlunya.
3. Gunakan bahasa Indonesia yang baik, sopan, dan mudah dipahami mahasiswa maupun masyarakat umum.

Aturan Grounding & Keamanan:
1. Jawab HANYA berdasarkan **[KONTEKS REFERENSI]** hasil pencarian TF-IDF dan **[RIWAYAT PERCAKAPAN]**. Jangan mengarang pasal, angka denda, jadwal, atau tarif yang tidak ada di konteks.
2. Jika di konteks terdapat rujukan resmi (seperti `PER-5/2024`, `PER-7/2024`, `PER-8/2024`, nomor pasal, atau URL resmi PKN STAN), sebutkan secara natural di bagian akhir atau di dalam penjelasan.
3. Jika pengguna menyapa atau berbasa-basi (misalnya "halo", "terima kasih", "siapa kamu"), balas dengan hangat dan tawarkan bantuan seputar PKN STAN.
4. Jika pertanyaan jelas di luar lingkup PKN STAN (misalnya cuaca, saham, kripto, resep masakan, zodiak), tolak dengan sopan dan jelaskan bahwa Wicak khusus membantu informasi seputar PKN STAN.
5. Jika informasi tidak ditemukan di konteks (atau terkait data terkunci seperti pricelist rinci Sewa Gedung), akui dengan jujur dan arahkan pengguna untuk menghubungi unit layanan resmi terkait.
"""

# ==============================================================================
# 2. PREPROCESSING & LOADER KNOWLEDGE BASE TF-IDF
# ==============================================================================
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    tokens = text.split()
    expanded = []
    for tok in tokens:
        if tok in STOPWORDS_ID:
            continue
        expanded.append(tok)
        if tok in SYNONYM_MAP:
            expanded.extend(SYNONYM_MAP[tok].split())
    return " ".join(expanded)


def _normalize_entry(raw: dict, idx: int) -> dict:
    judul = (
        raw.get("pertanyaan")
        or raw.get("question")
        or raw.get("judul")
        or raw.get("title")
        or raw.get("pasal")
        or f"Entri #{idx + 1}"
    )
    isi = (
        raw.get("jawaban")
        or raw.get("answer")
        or raw.get("isi")
        or raw.get("content")
        or raw.get("teks")
        or raw.get("text")
        or ""
    )
    sumber = (
        raw.get("sumber")
        or raw.get("source")
        or raw.get("referensi")
        or raw.get("reference")
        or raw.get("peraturan")
        or raw.get("url")
        or "Basis Pengetahuan PKN STAN"
    )
    kategori = (
        raw.get("kategori")
        or raw.get("category")
        or raw.get("topik")
        or raw.get("topic")
        or "Umum"
    )
    keywords = raw.get("keywords") or raw.get("kata_kunci") or raw.get("tags") or ""
    if isinstance(keywords, list):
        keywords = " ".join(str(k) for k in keywords)

    teks_pasal = raw.get("teks_pasal") or raw.get("pasal_lengkap") or isi
    combined_for_index = f"{judul} {kategori} {keywords} {isi}"
    return {
        "id": raw.get("id", idx + 1),
        "judul": str(judul).strip(),
        "isi": str(isi).strip(),
        "teks_pasal": str(teks_pasal).strip(),
        "sumber": str(sumber).strip(),
        "kategori": str(kategori).strip(),
        "processed": preprocess_text(combined_for_index),
    }


def load_kb(json_path: Path):
    if not json_path.exists():
        return [], None, None

    with json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        raw_entries = raw_data
    elif isinstance(raw_data, dict):
        raw_entries = []
        for key in ("entries", "data", "knowledge_base", "qa", "pasal", "items"):
            val = raw_data.get(key)
            if isinstance(val, list):
                raw_entries.extend(val)
        if not raw_entries:
            for val in raw_data.values():
                if isinstance(val, list):
                    raw_entries.extend(val)
    else:
        raw_entries = []

    entries = [_normalize_entry(item, i) for i, item in enumerate(raw_entries) if isinstance(item, dict)]
    corpus = [e["processed"] if e["processed"] else "kosong" for e in entries]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus) if corpus else None
    return entries, vectorizer, tfidf_matrix


KB_ENTRIES, TFIDF_VECTORIZER, TFIDF_MATRIX = load_kb(KB_PATH)


def retrieve_top_k(query: str, k: int = 5, threshold: float = 0.10):
    if not KB_ENTRIES or TFIDF_VECTORIZER is None or TFIDF_MATRIX is None:
        return []
    q_proc = preprocess_text(query)
    if not q_proc.strip():
        return []
    q_vec = TFIDF_VECTORIZER.transform([q_proc])
    scores = cosine_similarity(q_vec, TFIDF_MATRIX).flatten()
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score >= threshold:
            item = dict(KB_ENTRIES[idx])
            item["score"] = round(score, 4)
            results.append(item)
    return results

# ==============================================================================
# 3. INTEGRASI GEMINI API
# ==============================================================================
def generate_natural_reply(
    user_query: str,
    user_profile: str,
    chat_history: list,
    retrieved_docs: list,
    temperature: float = 0.3,
):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            "GEMINI_API_KEY belum diatur di server (.env atau environment variable).",
            None,
        )

    client = genai.Client(api_key=api_key)

    if retrieved_docs:
        context_block = "\n\n".join(
            f"[Referensi {i+1} | Skor TF-IDF: {doc['score']} | Sumber: {doc['sumber']}]\n"
            f"Topik: {doc['judul']}\n"
            f"Isi: {doc['isi']}"
            for i, doc in enumerate(retrieved_docs)
        )
    else:
        context_block = "Tidak ada dokumen di basis pengetahuan yang mencapai threshold kemiripan."

    recent_history = chat_history[-6:]
    history_block = "\n".join(
        f"{'Pengguna' if m.get('role') == 'user' else 'Wicak'}: {m.get('content', '')}"
        for m in recent_history
        if isinstance(m, dict)
    ) or "(Belum ada riwayat sebelumnya)"

    user_prompt = f"""[PROFIL PENGGUNA]: {user_profile}
[RIWAYAT PERCAKAPAN TERAKHIR]:
{history_block}

[KONTEKS REFERENSI]:
{context_block}

[PERTANYAAN TERBARU PENGGUNA]:
{user_query}"""

    last_error = None
    for model_name in ("gemini-3.5-flash-lite", "gemini-3.8-flash"):
        for _ in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=temperature,
                    ),
                )
                if response and response.text:
                    return response.text.strip(), model_name
            except Exception as exc:
                last_error = exc
                err_str = str(exc)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    time.sleep(1.2)
                    continue
                break

    return (
        f"Maaf, layanan AI sedang mengalami gangguan sementara. ({last_error})",
        None,
    )

# ==============================================================================
# 4. ROUTES FLASK (/ & /api/chat)
# ==============================================================================
@app.route("/", methods=["GET"])
def serve_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return send_from_directory(str(BASE_DIR), "index.html")
    return jsonify({
        "service": "Wicak PKN STAN Hybrid RAG API",
        "status": "running",
        "endpoints": {
            "health": "GET /api/health",
            "chat": "POST /api/chat",
        },
    })


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "kb_loaded": len(KB_ENTRIES),
        "gemini_key_configured": bool(os.getenv("GEMINI_API_KEY")),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Field 'query' wajib diisi."}), 400

    user_profile = str(payload.get("profile", "Civitas Akademika")).strip()
    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []

    top_k = int(payload.get("top_k", 5))
    threshold = float(payload.get("threshold", 0.10))
    temperature = float(payload.get("temperature", 0.3))

    # Gabungkan pertanyaan user sebelumnya jika query lanjutan sangat singkat
    retrieval_query = query
    if len(query.split()) <= 5 and history:
        prev_users = [m.get("content", "") for m in history if isinstance(m, dict) and m.get("role") == "user"]
        if prev_users:
            retrieval_query = f"{prev_users[-1]} {query}"

    matched_docs = retrieve_top_k(retrieval_query, k=top_k, threshold=threshold)
    reply_text, model_used = generate_natural_reply(
        user_query=query,
        user_profile=user_profile,
        chat_history=history,
        retrieved_docs=matched_docs,
        temperature=temperature,
    )

    references = [
        {
            "id": doc["id"],
            "judul": doc["judul"],
            "sumber": doc["sumber"],
            "kategori": doc["kategori"],
            "score": doc["score"],
            "teks_pasal": doc["teks_pasal"],
        }
        for doc in matched_docs
    ]

    return jsonify({
        "reply": reply_text,
        "references": references,
        "model_used": model_used,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)