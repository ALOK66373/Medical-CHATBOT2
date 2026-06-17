
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from torch import embedding
download_embeddings, 
from src.helper import download_embeddings, filter_to_minimal_docs, load_pdf_files, text_split_documents


extracted_data = load_pdf_files("data")

minimal_docs=filter_to_minimal_docs(extracted_data)

text_split_docs = text_split_documents(minimal_docs)

load_dotenv()

# Environment variables for Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY





embedding = download_embeddings()
# Save documents into local ChromaDB

persist_directory = "./chroma_db"

docsearch = Chroma.from_documents(
    documents=text_split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)

