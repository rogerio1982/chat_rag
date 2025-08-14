import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

API_URL = "http://localhost:8000/support?="


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    # ⚠️ CORREÇÃO AQUI
    # Acessa os dados JSON enviados pelo frontend.
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "Nenhuma mensagem fornecida."}), 400

    try:
        # A sua API FastAPI espera o campo 'text' como form-data.
        # Por isso, vamos usar o dicionário `data` para o `requests`.
        payload = {'text': user_message}

        response = requests.post(API_URL, data=payload)
        response.raise_for_status()

        api_response = response.json()

        answer = api_response.get("answer", "Desculpe, não consegui obter uma resposta.")

        return jsonify({"answer": answer})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Erro ao conectar com a API: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)