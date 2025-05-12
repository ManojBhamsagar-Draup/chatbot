import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings # Old
from langchain_huggingface import HuggingFaceEmbeddings  # New
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "documents/"
DB_FAISS_PATH = "faiss_index/"


def create_vector_db():
    print(f"Loading documents from {DOCS_PATH}...")
    doc_files = [os.path.join(DOCS_PATH, f) for f in os.listdir(DOCS_PATH) if f.endswith('.txt')]
    all_documents = []
    for doc_file in doc_files:
        loader = TextLoader(doc_file, encoding='utf-8')
        all_documents.extend(loader.load())

    if not all_documents:
        print("No documents found. Please check the DOCS_PATH.")
        return

    print(f"Loaded {len(all_documents)} document sections initially.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = text_splitter.split_documents(all_documents)
    print(f"Split into {len(texts)} chunks.")

    print("Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print("Creating FAISS index...")
    db = FAISS.from_documents(texts, embeddings)

    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)

    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index created and saved to {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_db()