import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

def test_mistral_conn():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("MISTRAL_API_KEY manquante")
        return
    
    llm = ChatMistralAI(mistral_api_key=api_key, model="mistral-tiny")
    try:
        res = llm.invoke("Hello, are you working?")
        print(f"Mistral response: {res.content}")
    except Exception as e:
        print(f"Mistral error: {e}")

if __name__ == "__main__":
    test_mistral_conn()
