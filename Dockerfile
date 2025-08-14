# ===== Base PyTorch CPU enxuta =====
FROM pytorch/pytorch:2.2.0-cpu

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# Copia apenas o requirements.txt
COPY requirements.txt .

# Instala apenas dependências de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        libsm6 \
        libxext6 \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Instala dependências Python, exceto PyTorch (já incluso na imagem base)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --no-deps \
    && rm -rf /root/.cache/pip

# Copia o código da aplicação
COPY . .

# Expõe porta FastAPI
EXPOSE 8000

# Comando padrão
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
