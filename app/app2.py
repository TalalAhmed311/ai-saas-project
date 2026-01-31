import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from dotenv import load_dotenv
from app.prompts import RAG_PROMPT
import shutil
from pathlib import Path
import asyncio

load_dotenv()

app = FastAPI()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_collection2")

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

# Initialize Async Qdrant Client
async_client = AsyncQdrantClient(url=QDRANT_URL)

async def ensure_collection_exists():
    from qdrant_client.http import models
    try:
        collections = await async_client.get_collections()
        exists = any(c.name == COLLECTION_NAME for c in collections.collections)
        if not exists:
            print(f"Creating collection: {COLLECTION_NAME}")
            vector_size = 1536 if MODEL_PROVIDER == "openai" else 768
            await async_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            print(f"Collection {COLLECTION_NAME} created")
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")

@app.on_event("startup")
async def startup_event():
    await ensure_collection_exists()

@app.get("/")
async def read_root():
    return {"status": "Async RAG API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_path = Path(f"temp_async_{file.filename}")
    
    # Async file writing
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run blocking PDF loading/splitting in a thread pool
        loader = PyPDFLoader(str(temp_path))
        documents = await asyncio.to_thread(loader.load)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
        print(f"Successfully split {len(documents)} documents into {len(chunks)} chunks (Async)")
        
        
        # 3. Async upload
        # We use the client directly because langchain-qdrant's aadd_documents 
        # has a bug where it doesn't await the upsert call.
        from qdrant_client.http import models
        import uuid
        
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings.embed_query(chunk.page_content),
                payload={
                    "page_content": chunk.page_content,
                    **chunk.metadata
                }
            ) for chunk in chunks
        ]
        
        await async_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        print(f"Successfully processed {len(chunks)} chunks from {file.filename} (Async)")
        return {"message": f"Successfully processed {len(chunks)} chunks from {file.filename} (Async)"}

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
        # 2. Search for similar documents
        # We use the client directly to avoid a bug in langchain-qdrant's asimilarity_search
        query_embedding = await embeddings.aembed_query(text)
        search_result = await async_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=3,
            with_payload=True
        )
        print(search_result)
        
        from langchain_core.documents import Document
        docs = [
            Document(page_content=point.payload.get("page_content", ""), metadata=point.payload)
            for point in search_result.points
        ]
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = RAG_PROMPT.format(context=context, question=text)

        # Async LLM invocation
        llm = get_llm()
        response = await llm.ainvoke(prompt)

        return {
            "query": text,
            "response": response.content,
            "sources": [doc.metadata for doc in docs]
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

