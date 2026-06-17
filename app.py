from functools import lru_cache
from flask import Flask, render_template, request, session
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from src.helper import download_hugging_face_embeddings, filter_to_minimal_docs, load_pdf_files, text_split_documents
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from src.prompt import *
import os
import shutil
import uuid
import logging


app = Flask(__name__)

load_dotenv()

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "change-this-in-production"
)

logging.basicConfig(level=logging.INFO)

# Environment variables for Gemini
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if API_KEY:
    os.environ["GOOGLE_API_KEY"] = API_KEY
    os.environ["GEMINI_API_KEY"] = API_KEY

chat_histories = {}


def get_api_error_message(exc: Exception) -> str:
    message = str(exc).lower()

    if any(
        term in message
        for term in (
            "quota",
            "resource_exhausted",
            "429",
            "rate limit",
            "exceeded your current quota",
        )
    ):
        return (
            "The chatbot could not respond because the Gemini API quota has been "
            "exceeded. Please wait a bit or use a billing-enabled API key."
        )

    if any(
        term in message
        for term in (
            "api key",
            "authentication",
            "unauthorized",
            "invalid key",
            "permission denied",
        )
    ):
        return (
            "The chatbot could not respond because the API key is missing or "
            "invalid. Please check the environment configuration."
        )

    return (
        "Sorry, the chatbot could not respond right now. Please try again in a "
        "few minutes."
    )

# ------------------- NEW: lazy + cached setup -------------------

@lru_cache(maxsize=1)
def get_embeddings():
    return download_hugging_face_embeddings()


@lru_cache(maxsize=1)
def get_docsearch():
    persist_directory = os.path.abspath("./chroma_db")
    embeddings = get_embeddings()

    if os.path.isdir(persist_directory) and os.listdir(persist_directory):
        try:
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
        except Exception:
            pass

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

    extracted_data = load_pdf_files("data")
    minimal_docs = filter_to_minimal_docs(extracted_data)
    text_split_docs = text_split_documents(minimal_docs)

    return Chroma.from_documents(
        documents=text_split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


@lru_cache(maxsize=1)
def get_retriever():
    docsearch = get_docsearch()
    return docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.3,
        top_p=0.95,
        top_k=40,
    )


@lru_cache(maxsize=1)
def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


@lru_cache(maxsize=1)
def get_question_answer_chain():
    return create_stuff_documents_chain(get_llm(), get_prompt())


@lru_cache(maxsize=1)
def get_rag_chain():
    return create_retrieval_chain(get_retriever(), get_question_answer_chain())


def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


def get_conversational_rag_chain():
    return RunnableWithMessageHistory(
        get_rag_chain(),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# ------------------- OLD eager setup (kept for reference) -------------------
# embeddings = download_hugging_face_embeddings()
#
# persist_directory = os.path.abspath("./chroma_db")
#
# extracted_data = load_pdf_files("data")
# minimal_docs = filter_to_minimal_docs(extracted_data)
# text_split_docs = text_split_documents(minimal_docs)
#
#
# def build_docsearch():
#     if os.path.exists(persist_directory):
#         shutil.rmtree(persist_directory, ignore_errors=True)
#
#     try:
#         return Chroma.from_documents(
#             documents=text_split_docs,
#             embedding=embeddings,
#             persist_directory=persist_directory,
#         )
#     except Exception as e:
#         message = str(e).lower()
#         if "hnsw" not in message and "compaction" not in message and "internalerror" not in message:
#             raise
#
#         if os.path.exists(persist_directory):
#             shutil.rmtree(persist_directory, ignore_errors=True)
#
#         return Chroma.from_documents(
#             documents=text_split_docs,
#             embedding=embeddings,
#             persist_directory=persist_directory,
#         )
#
#
# docsearch = build_docsearch()
#
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
#
#
#
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash",
#     temperature=0.3,
#     top_p=0.95,
#     top_k=40,
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#     ]
# )
#
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)
#
# chat_histories = {}
#
#
# def get_session_history(session_id: str):
#     if session_id not in chat_histories:
#         chat_histories[session_id] = InMemoryChatMessageHistory()
#     return chat_histories[session_id]
#
#
# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
# )


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question."

    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())

    try:
        conversational_rag_chain = get_conversational_rag_chain()
        response = conversational_rag_chain.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session["session_id"]}},
        )
        return str(response.get("answer", "I couldn't generate a response right now."))
    except Exception as e:
        logging.exception("Chat request failed")
        return get_api_error_message(e)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)