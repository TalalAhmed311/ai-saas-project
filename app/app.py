import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from app.prompts import RAG_PROMPT
import shutil
from pathlib import Path

load_dotenv()

app = FastAPI()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()  # 'gemini' or 'openai'
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection")

def get_embeddings():
    if MODEL_PROVIDER == "openai":
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def get_llm():
    if MODEL_PROVIDER == "openai":
        return ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Initialize Embeddings
embeddings = get_embeddings()

# Initialize Qdrant Client
client = QdrantClient(url=QDRANT_URL)

def ensure_collection_exists():
    from qdrant_client.http import models
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            print(f"Creating collection: {COLLECTION_NAME}")
            # Determine vector size based on provider
            vector_size = 1536 if MODEL_PROVIDER == "openai" else 768
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")

# Ensure collection is created on startup
ensure_collection_exists()

@app.get("/")
def read_root():
    return {"status": "RAG API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save file temporarily
    temp_path = Path(f"temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print(temp_path)
        print(file.filename)
        # 1. Parse PDF
        loader = PyPDFLoader(str(temp_path))
        documents = loader.load()
        print(len(documents))

        # 2. Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        print(len(chunks))
        print(chunks[0])
        print(chunks[0].page_content)
        print(chunks[0].metadata)
        print(QDRANT_URL)
        print(COLLECTION_NAME)
        print(embeddings)
        print(embeddings.embed_documents(["Hello, world!"]))
        # 3. Create embeddings and upload to Qdrant using the existing client
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        vector_store.add_documents(chunks)

        return {"message": f"Successfully processed {len(chunks)} chunks from {file.filename}"}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            os.remove(temp_path)

@app.post("/query")
async def query_rag(text: str):
    try:
        # 1. Initialize Vector Store using the existing client
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )

        # 2. Search for similar documents
        docs = vector_store.similarity_search(text, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 3. Format prompt
        prompt = RAG_PROMPT.format(context=context, question=text)

        # 4. Generate response using selected LLM
        llm = get_llm()
        response = llm.invoke(prompt)

        return {
            "query": text,
            "response": response.content,
            "sources": [doc.metadata for doc in docs]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
