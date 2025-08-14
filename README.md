# FastAPI Multimodal RAG API com Gemini

Este projeto inclui uma API **FastAPI** para **RAG (Retrieval-Augmented Generation)** multimodal com o modelo **Gemini**, e um chat em **Flask** para interagir com ela.  
A API processa **texto**, **áudio** e **imagens** (via OCR) com base em documentos de uma base de conhecimento.

---

## 📋 Pré-requisitos

Para o projeto funcionar, você precisa do **Python 3.8+** e de duas ferramentas externas instaladas no seu sistema:

- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** → Essencial para ler texto de imagens.  
- **[FFmpeg](https://ffmpeg.org/download.html)** → Necessário para processar arquivos de áudio.

---

## ⚙️ Configuração e Instalação

### 1️⃣ Clone o projeto e crie um ambiente virtual
```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
