"""
FastAPI Multimodal RAG API com Gemini
-------------------------------------
Endpoints:
POST /support        -> aceita texto + imagem opcional
POST /support/audio  -> aceita áudio (.mp3/.wav) + imagem opcional
"""

import os
import io
import uuid
import tempfile
import json
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PIL import Image
import pytesseract
import PyPDF2
import whisper
from gtts import gTTS
from sentence_transformers import SentenceTransformer
import faiss
import re

# Gemini
import google.generativeai as genai

# Configura a API key do Gemini
genai.configure(api_key="keyhere")


# Diretórios
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

# Modelos
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

app = FastAPI(title="Multimodal RAG API - Gemini")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Globais
_whisper_model = None
_embeddings_model = None
_faiss_index = None
_documents = []
_texts = []

# --- Funções utilitárias ---

def chunk_text(text, max_len=300):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model

def load_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embeddings_model

def index_exists():
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss")) and os.path.exists(os.path.join(INDEX_DIR, "meta.json"))

def save_index(index, metas, texts):
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    data = {"metas": metas, "texts": texts}
    with open(os.path.join(INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_index():
    global _faiss_index, _documents, _texts
    idx_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "meta.json")

    # 1. Carrega o índice do FAISS
    _faiss_index = faiss.read_index(idx_path)

    # 2. Carrega os metadados
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Adiciona uma verificação para garantir que 'data' é um dicionário
    if isinstance(data, dict):
        _documents = data.get("metas", [])
        _texts = data.get("texts", [])
    else:
        # Se 'data' não for um dicionário, algo está errado no arquivo.
        # Reinicializamos as variáveis para evitar mais erros.
        print("Erro: O arquivo meta.json não está no formato esperado (dicionário).")
        _documents = []
        _texts = []

    return _faiss_index

def build_index_from_kb():
    global _faiss_index, _documents, _texts
    emb = load_embeddings_model()
    texts, metas = [], []
    for root, _, files in os.walk(KB_DIR):
        for fn in files:
            path = os.path.join(root, fn)
            ext = fn.lower().split('.')[-1]
            text = ""
            try:
                if ext in ("txt", "md"):
                    with open(path, "r", encoding="utf-8", errors='ignore') as f:
                        text = f.read()
                elif ext == "pdf":
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "\n".join([p.extract_text() or "" for p in reader.pages])
            except Exception as e:
                print(f"Erro lendo {path}: {e}")
                continue
            if text.strip():
                for idx, chunk in enumerate(chunk_text(text)):
                    texts.append(chunk)
                    metas.append({"source": fn, "path": path, "chunk_id": idx})
    if not texts:
        return None
    embeddings = emb.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    save_index(index, metas, texts)
    _faiss_index = index
    _documents = metas
    _texts = texts
    return index

def retrieve(query: str, top_k=TOP_K):
    emb = load_embeddings_model()
    global _faiss_index, _documents, _texts
    if _faiss_index is None:
        if index_exists():
            load_index()
        else:
            build_index_from_kb()
    if _faiss_index is None:
        return []
    q_emb = emb.encode([query], convert_to_numpy=True)
    _, I = _faiss_index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(_documents):
            continue
        meta = _documents[idx]
        snippet = _texts[idx][:1000] if idx < len(_texts) else ""
        results.append({"source": meta['source'], "snippet": snippet})
    return results

def generate_answer(question: str, context_docs: List[dict]) -> str:
    """
    Gera uma resposta técnica e amigável usando Gemini LLM,
    baseada em documentos de suporte.
    """
    # Se não há contexto, resposta genérica e profissional
    if not context_docs:
        return (
            "Olá! Não encontrei informações relevantes nos meus documentos para responder a sua pergunta. "
            "Por favor, reformule a questão ou forneça mais detalhes."
        )

    # Monta o contexto para o prompt
    context_text = "\n\n".join([f"[{d['source']}]: {d['snippet']}" for d in context_docs])

    # Prompt para o Gemini com as novas instruções
    prompt = (
        "Você é um **assistente técnico de suporte**. Sua função é responder a perguntas de forma clara, "
        "precisa e amigável, como se estivesse ajudando um colega de trabalho ou um cliente. "
        "Utilize a informação fornecida no contexto abaixo para formular sua resposta.\n\n"
        "**Instruções:**\n"
        "1.  **Seja direto e objetivo:** A resposta deve ir direto ao ponto, sem rodeios.\n"
        "2.  **Mantenha um tom profissional e prestativo:** Use uma linguagem acessível, evite gírias e jargões complexos, a menos que sejam estritamente necessários.\n"
        "3.  **Cite as fontes quando possível:** Se a resposta for baseada em uma fonte específica, mencione-a (ex: 'De acordo com o manual, ...').\n"
        "4.  **Se a informação for insuficiente:** Se o contexto não tiver dados suficientes para uma resposta completa, diga de forma educada que a informação não foi encontrada no material de referência. Não invente ou presuma respostas.\n"
        "5.  **Formate a resposta:** Use listas, negrito ou itálico para organizar a informação e torná-la mais fácil de ler, especialmente em instruções passo a passo.\n\n"
        f"**Contexto:**\n{context_text}\n\n**Pergunta:** {question}\n**Resposta:**"
    )

    # Chamada Gemini usando Client
    # NOTE: O seu código tem uma chamada a `client.generate_text`, que é de uma API mais antiga.
    # A API do Gemini Pro 1.5 usa `genai.GenerativeModel`. Recomendo atualizar para a versão mais recente.
    # Exemplo de como ficaria com a nova API:
    # model = genai.GenerativeModel('gemini-1.5-pro-latest')
    # response = model.generate_content(prompt)
    # return response.text

    # Mantendo a sua chamada original por compatibilidade
    # Use genai.GenerativeModel para a API mais recente
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    return response.text


def transcribe_audio(file_path: str) -> str:
    model = load_whisper_model()
    res = model.transcribe(file_path)
    return res.get('text', '').strip()

def ocr_image(image_path: str) -> str:
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""

def text_to_speech(text: str, out_path: str) -> str:
    tts = gTTS(text=text, lang='es')
    tts.save(out_path)
    return out_path

# --- Schemas ---
class SupportResponse(BaseModel):
    transcription: Optional[str]
    ocr_text: Optional[str]
    answer: str
    audio_url: Optional[str]
    source_documents: List[str]

# --- Endpoints ---

@app.post('/support', response_model=SupportResponse)
async def support_endpoint(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    if not text and not image:
        raise HTTPException(status_code=400, detail="Forneça 'text' ou 'image'.")

    ocr_text = None
    if image:
        suffix = os.path.splitext(image.filename)[1].lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await image.read())
        tmp.close()
        ocr_text = ocr_image(tmp.name)
        os.unlink(tmp.name)

    query = text or ""
    if ocr_text:
        query += "\n" + ocr_text

    docs = retrieve(query)
    sources = [d['source'] for d in docs]
    answer = generate_answer(query, docs)

    mp3_name = f"response_{uuid.uuid4().hex}.mp3"
    mp3_path = os.path.join(STATIC_DIR, mp3_name)
    try:
        text_to_speech(answer, mp3_path)
        audio_url = f"/static/{mp3_name}"
    except Exception:
        audio_url = None

    return SupportResponse(transcription=None, ocr_text=ocr_text, answer=answer, audio_url=audio_url, source_documents=sources)

@app.post('/support/audio', response_model=SupportResponse)
async def support_audio_endpoint(
    audio: UploadFile = File(...),
    image: Optional[UploadFile] = File(None)
):
    suffix = os.path.splitext(audio.filename)[1].lower()
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_audio.write(await audio.read())
    tmp_audio.close()
    transcription = transcribe_audio(tmp_audio.name)
    os.unlink(tmp_audio.name)

    ocr_text = None
    if image:
        img_suffix = os.path.splitext(image.filename)[1].lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=img_suffix)
        tmp.write(await image.read())
        tmp.close()
        ocr_text = ocr_image(tmp.name)
        os.unlink(tmp.name)

    query = transcription or ""
    if ocr_text:
        query += "\n" + ocr_text

    docs = retrieve(query)
    sources = [d['source'] for d in docs]
    answer = generate_answer(transcription, docs)

    mp3_name = f"response_{uuid.uuid4().hex}.mp3"
    mp3_path = os.path.join(STATIC_DIR, mp3_name)
    try:
        text_to_speech(answer, mp3_path)
        audio_url = f"/static/{mp3_name}"
    except Exception:
        audio_url = None

    return SupportResponse(transcription=transcription, ocr_text=ocr_text, answer=answer, audio_url=audio_url, source_documents=sources)

@app.get('/health')
async def health_check():
    return {"status": "ok"}


