import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from src.vector_store import load_vector_store

# Chargement des variables d'environnement
load_dotenv()

def get_chatbot_chain():
    """
    Initialise et retourne la chaîne RAG pour le chatbot.
    """
    # 1. Chargement du vector store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Initialisation du modèle Mistral
    llm = ChatMistralAI(
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        model=os.getenv("MISTRAL_MODEL", "mistral-tiny"),
        temperature=0.2
    )

    # 3. Définition du prompt template
    system_prompt = (
        "Vous êtes un assistant spécialisé dans les événements culturels à Lille et dans les Hauts-de-France. "
        "Utilisez les éléments de contexte suivants pour répondre à la question de l'utilisateur. "
        "Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse. "
        "Soyez amical et donnez des détails précis sur les lieux, dates et descriptions des événements mentionnés dans le contexte. "
        "S'il y a plusieurs événements, présentez-les sous forme de liste à puces."
        "\n\n"
        "Contexte :\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 4. Création de la chaîne de documents
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 5. Création de la chaîne de récupération
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def ask_chatbot(query, rag_chain):
    """
    Pose une question au chatbot et retourne la réponse.
    """
    response = rag_chain.invoke({"input": query})
    return response["answer"]

if __name__ == "__main__":
    # Test rapide du chatbot
    print("Initialisation du chatbot...")
    chain = get_chatbot_chain()
    
    query = "Quels sont les événements musicaux prévus à Lille ?"
    print(f"\nQuestion : {query}")
    answer = ask_chatbot(query, chain)
    print(f"\nRéponse du chatbot :\n{answer}")
