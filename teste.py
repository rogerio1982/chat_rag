
from google import genai

# Substitua 'SUA_CHAVE_AQUI' pela sua chave da API do Gemini
client = genai.Client(api_key="AIzaSyDDAXLbSQBeC0JGNdCAafQ9Af0WPAIy0Yo")

# Defina o prompt para o modelo
prompt = "ola tudo bem"

# Gere a resposta
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# Exiba a resposta gerada
print("Resposta:", response.text)
