import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = r"data\dsa_notes.pdf"


import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone


import time

def embed_in_batches(embeddings, texts, batch_size=25, sleep_sec=1):
    all_vectors = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]

        try:
            print(f"🧠 Embedding batch {i//batch_size + 1} / {(total + batch_size - 1)//batch_size}")
            vectors = embeddings.embed_documents(batch)
            all_vectors.extend(vectors)

        except Exception as e:
            print(f"⚠️ Embedding failed at batch {i}. Retrying...")
            time.sleep(5)
            vectors = embeddings.embed_documents(batch)
            all_vectors.extend(vectors)

        time.sleep(sleep_sec)  # rate-limit safety

    return all_vectors

def load_pdf(path):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"page": i + 1}
                    )
                )
    return docs

def ingest():
    print(" STEP 1: Loading PDF...")
    documents = load_pdf(PDF_PATH)
    print(f" Loaded {len(documents)} pages")

    print(" STEP 2: Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250
    )
    chunks = splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks")

    print(" STEP 3: Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    print(" Embedding model ready")
    print(" STEP 4: Generating embeddings (batched)...")

    texts = [doc.page_content for doc in chunks]

    print(" STEP 4: Generating embeddings (micro-batched)...")
    vectors = embed_in_batches(embeddings, texts)
    print(f" Generated {len(vectors)} embeddings")

    print(" STEP 5: Uploading embeddings to Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    upserts = []
    for i, vector in enumerate(vectors):
        upserts.append(
            (
                f"chunk-{i}",
                vector,
                {"text": texts[i]}
            )
            )
        index.upsert(vectors=upserts)
    print("🎉 INGESTION COMPLETE!")






if __name__ == "__main__":
    ingest()
