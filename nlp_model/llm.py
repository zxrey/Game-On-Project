import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def mejorar_consulta(consulta):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # modelo más grande y preciso
            messages=[{"role": "user", "content": f"""You are a video game expert.
        Given this search query: '{consulta}'

        1. Translate the query to English if it is in another language
        2. Write the translated query
        2. Extract the main theme/enemy/setting keywords
        3. Repeat those keywords 3 times to give them more weight
        4. Add 5 related gaming terms in English

        Example:
        'kill horde of demons' → 'demons demons demons, kill horde of demons,
        demon slayer, hellish, gore, FPS, shooter'

        Return only the words, no explanation."""}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception:
        # Si Groq falla, seguimos la búsqueda con la consulta original sin mejorar
        return consulta


def generar_descripcion(game, consulta):
    # Genera una descripción personalizada explicando por qué el juego es una buena recomendación
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user",
                       "content": f"En 2 oraciones explica por qué el juego '{game['name']}' ({game['genre']}) es una buena recomendación para alguien que busca: {consulta}"}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception:
        return "No se pudo generar una descripción para este juego en este momento."
