# RAG Application with FastAPI, Gemini, and Qdrant

A simple Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents, process them into vector embeddings using Google Gemini, and query them using a natural language interface.

## Features
- **PDF Upload**: Parse and chunk PDF files using LangChain.
- **Vector Search**: Store and retrieve document context using Qdrant.
- **Gemini AI**: Use Google's Gemini Pro for embeddings and response generation.
- **FastAPI**: Modern, high-performance web API.

## Prerequisites
- Docker and Docker Compose
- Python 3.10+
- Google AI API Key (Gemini)

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd ai-saas-project
```

### 2. Environment Configuration
Create a `.env` file in the root directory (or update the existing one):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=rag_collection
```

### 3. Start Qdrant (Vector Database)
Run the following command to start the Qdrant container:
```bash
docker-compose up -d
```

### 4. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirement.txt
```

### 5. Run the Application
Start the FastAPI server:
```bash
uvicorn app.app:app --reload
```

## API Usage

### Upload a PDF
- **Endpoint**: `POST /upload`
- **Body**: `file` (form-data)
- **Description**: Uploads a PDF, chunks it, and stores embeddings in Qdrant.

### Query the RAG
- **Endpoint**: `POST /query`
- **Query Params**: `text` (string)
- **Description**: Searches for relevant context in the uploaded documents and generates an answer using Gemini.

## API Documentation
Once the server is running, you can access the interactive Swagger UI at:
[http://localhost:8000/docs](http://localhost:8000/docs)

