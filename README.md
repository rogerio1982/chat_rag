# FastAPI Multimodal RAG API with Gemini

This project includes a **FastAPI** for **RAG (Retrieval-Augmented Generation)** multimodal using the **Gemini** model, and a **Flask** chat interface to interact with it.  
The API processes **text**, **audio**, and **images** (via OCR) based on documents from a knowledge base.


---

## 📋 Prerequisites

To run this project, you will need **Python 3.8+** and two external tools installed on your system:

- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** → Essencial para ler texto de imagens.  
- **[FFmpeg](https://ffmpeg.org/download.html)** → Necessário para processar arquivos de áudio.

---

## ⚙️ ⚙️ Setup and Installation

### 11 Clone the repository and create a virtual environment
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
---

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
---

### 2 Install dependencies
pip install -r requirements.txt
---

### 3 Configure your Gemini API key (LLM)
Get your key from Google AI Studio.
Replace "YOUR_KEY_HERE" in the main.py file.

### 4 How to Run
Você precisa rodar **dois servidores** em terminais separados.

### 5  Start the FastAPI server
```bash
uvicorn main:app --reload --port 8000
```
### 6 Start the Flask server (in another terminal)
```bash
uvicorn main:app --reload --port 8000
```
### 6 Accessing the Chat
```bash
http://127.0.0.1:5000
```
### 6 Or test using Postman:
```bash
http://127.0.0.1:5000
