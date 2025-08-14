"""
FastAPI Multimodal RAG API
-------------------------
Endpoints:
POST /support        -> aceita texto + imagem opcional
POST /support/audio  -> aceita áudio (.mp3/.wav) + imagem opcional
"""
import re
import os
import io
import uuid
import tempfile
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Modelos ML
import whisper
from gtts import gTTS
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
os.makedirs(KB_DIR, exist_ok=True)
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")
os.makedirs(INDEX_DIR, exist_ok=True)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_LLM_MODEL = os.environ.get("HF_LLM_MODEL", "gpt2")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

app = FastAPI(title="Multimodal RAG API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_whisper_model = None
_embeddings_model = None
_faiss_index = None
_documents = []
_texts = []
_llm_pipeline = None

def chunk_text(text, max_len=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len):
        chunk = " ".join(words[i:i+max_len])
        chunks.append(chunk)
    return chunks

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
    _faiss_index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Detecta se é lista ou dict
    if isinstance(data, dict):
        _documents = data.get("metas", [])
        _texts = data.get("texts", [])
    elif isinstance(data, list):
        _documents = data
        _texts = []  # Sem textos carregados
    else:
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
                chunks = chunk_text(text, max_len=300)
                for idx, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metas.append({"source": fn, "path": path, "chunk_id": idx})
    if not texts:
        return None
    embeddings = emb.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    save_index(index, metas,texts)
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
        snippet = ""
        if idx < len(_texts):
            snippet = _texts[idx][:1000]
        results.append({"source": meta['source'], "snippet": snippet})
    return results

def load_llm():
    global _llm_pipeline
    if _llm_pipeline is None:
        _llm_pipeline = pipeline("text-generation", model=HF_LLM_MODEL, tokenizer=HF_LLM_MODEL)
    return _llm_pipeline

def generate_answer(question: str, context_docs: List[dict]) -> str:
    # Se não há contexto, retorna resposta padrão
    if not context_docs or all(not d['snippet'].strip() for d in context_docs):
        return "Por favor verifique si su tarjeta gráfica está bien conectada."

    prompt = (
        "Usted es un asistente técnico. "
        "Use SOLO el contexto abajo para responder. "
        "Responda claro y directo, sin inventar información.\n\n"
    )
    for i, d in enumerate(context_docs):
        prompt += f"[DOC {i+1} - {d['source']}]:\n{d['snippet']}\n\n"
    prompt += f"Pregunta: {question}\nRespuesta:"

    llm = load_llm()
    out = llm(prompt, max_length=256, do_sample=False, num_return_sequences=1)
    text = out[0]['generated_text']

    if "Respuesta:" in text:
        answer = text.split("Respuesta:")[-1].strip()
    else:
        answer = text.strip()

    answer = re.sub(r'\b(\w+)( \1){3,}', r'\1', answer)
    return answer


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

class SupportResponse(BaseModel):
    transcription: Optional[str]
    ocr_text: Optional[str]
    answer: str
    audio_url: Optional[str]
    source_documents: List[str]

@app.post('/support', response_model=SupportResponse)
async def support_endpoint(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    if not text and not image:
        raise HTTPException(status_code=400, detail="Forneça 'text' ou 'image'.")
    transcription = None
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
    return SupportResponse(transcription=transcription, ocr_text=ocr_text, answer=answer, audio_url=audio_url, source_documents=sources)

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
#AIzaSyDDAXLbSQBeC0JGNdCAafQ9Af0WPAIy0Yo