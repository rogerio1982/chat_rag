FastAPI Multimodal RAG API com Gemini
Este projeto inclui uma API FastAPI para RAG (Retrieval-Augmented Generation) multimodal com o modelo Gemini, e um chat em Flask para interagir com ela. A API processa texto, áudio e imagens (via OCR) com base em documentos de uma base de conhecimento.

Pré-requisitos
Para o projeto funcionar, você precisa do Python 3.8+ e de duas ferramentas externas instaladas no seu sistema:

Tesseract OCR: Essencial para ler texto de imagens. Guia de instalação.

FFmpeg: Necessário para processar arquivos de áudio.

Configuração e Instalação
Clone o projeto e crie um ambiente virtual.

Bash

python -m venv venv
venv\Scripts\activate  # ou source venv/bin/activate no macOS/Linux
Instale as dependências do requirements.txt.

Bash

pip install -r requirements.txt
Configure sua chave do Gemini.

Pegue sua chave no Google AI Studio.

Substitua "SUA_CHAVE_AQUI" no arquivo main.py.

Adicione seus documentos (.txt, .md, .pdf) na pasta knowledge_base para que o assistente possa consultá-los.

Como Rodar
Você precisa rodar dois servidores em terminais separados.

Inicie a API FastAPI

Bash

uvicorn main:app --reload --port 8000
Inicie o servidor Flask (em outro terminal)

Bash

python app.py
Depois, acesse http://127.0.0.1:5000 no seu navegador para usar o chat.
