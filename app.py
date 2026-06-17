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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embeddings = download_hugging_face_embeddings()

persist_directory = os.path.abspath("./chroma_db")

extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_split_docs = text_split_documents(minimal_docs)


def build_docsearch():
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

    try:
        return Chroma.from_documents(
            documents=text_split_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
    except Exception as e:
        message = str(e).lower()
        if "hnsw" not in message and "compaction" not in message and "internalerror" not in message:
            raise

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory, ignore_errors=True)

        return Chroma.from_documents(
            documents=text_split_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )


docsearch = build_docsearch()

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})



llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.3,
    top_p=0.95,
    top_k=40,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

chat_histories = {}


def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


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
        response = conversational_rag_chain.invoke(
            {"input": msg},
            config={"configurable": {"session_id": session["session_id"]}},
        )
        return str(response.get("answer", "I couldn't generate a response right now."))
    except Exception as e:
        logging.exception("Chat request failed")
        return (
            "Sorry, the chatbot could not respond right now. "
            f"Details: {e}"
        )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)