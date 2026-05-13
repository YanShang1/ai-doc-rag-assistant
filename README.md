# AI Document Q&A Assistant

A Streamlit-based Retrieval-Augmented Generation app that lets users upload PDF documents and ask questions grounded in the uploaded content.

## Features

- Upload PDF files
- Extract text from PDFs
- Split documents into chunks
- Generate embeddings
- Store document chunks in ChromaDB
- Retrieve relevant chunks for each question
- Generate LLM answers based on retrieved context
- Show source page references and relevant excerpts

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- OpenAI API
- PyMuPDF

## Project Structure

```text
ai-doc-rag-assistant/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   └── config.py
├── data/
└── assets/
```

## Installation

```bash
git clone https://github.com/your-username/ai-doc-rag-assistant.git
cd ai-doc-rag-assistant

python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Then add your OpenAI API key:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

## Run

```bash
streamlit run app.py
```

## How It Works

1. User uploads a PDF.
2. The app extracts text page by page.
3. Text is split into overlapping chunks.
4. Each chunk is embedded using OpenAI embeddings.
5. Chunks are stored in ChromaDB.
6. User asks a question.
7. The app retrieves the most relevant chunks.
8. The LLM answers using only the retrieved document context.
9. Sources are displayed below the answer.

## Resume Description

**AI Document Q&A Assistant | Python, Streamlit, LangChain, ChromaDB, OpenAI API**

- Built a Retrieval-Augmented Generation application that allows users to upload PDFs and ask natural language questions.
- Implemented document parsing, text chunking, embedding generation, vector storage, semantic retrieval, and source-grounded answer generation.
- Integrated ChromaDB for vector similarity search and displayed retrieved source excerpts to improve transparency.
- Designed an interactive Streamlit interface for document upload, processing, and real-time AI question answering.
