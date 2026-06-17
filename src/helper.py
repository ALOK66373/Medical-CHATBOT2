




from functools import lru_cache
from typing import List
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------- NEW: cached embedding loader -------------------


def resolve_data_dir(data="data"):
    data_dir = Path(data)

    if data_dir.is_absolute():
        if data_dir.exists():
            return data_dir.resolve()
        raise FileNotFoundError(
            f"Could not locate data directory: {data_dir}"
        )

    cwd = Path.cwd()

    if (cwd / data).exists():
        return (cwd / data).resolve()

    for parent in cwd.parents:
        if (parent / data).exists():
            return (parent / data).resolve()

    fallback = (
        Path.home()
        / "OneDrive"
        / "Documents"
        / "medical-chatbot"
        / data
    )

    if fallback.exists():
        return fallback.resolve()

    raise FileNotFoundError(
        f"Could not locate data directory: {data}"
    )


def load_pdf_files(data="data"):
    data_dir = resolve_data_dir(data)

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


def filter_to_minimal_docs(documents: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in documents:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs


#Split the documents into smaller chunks

def text_split_documents(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.create_documents([doc.page_content for doc in documents])
    return split_docs



from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def download_hugging_face_embeddings():
    return download_embeddings()

# ------------------- OLD eager embedding setup (kept for reference) -------------------
# embedding = download_embeddings()
