# RAG Chat with PDF Analysis

## Overview
This project is a **Retrieval-Augmented Generation (RAG) chat application** that allows users to upload PDFs and interact with an AI assistant to extract and retrieve information from the uploaded documents. The AI assistant is powered by **LangChain**, **Groq AI**, and **Hugging Face Embeddings**.

## Features
- **PDF Upload & Storage**: Upload multiple PDF files for analysis.
- **Vector Store Creation**: The system processes PDFs, extracts text, and stores them as embeddings for efficient retrieval.
- **AI-Powered Chat**: Users can ask questions about uploaded documents, and the AI will provide relevant responses based on the extracted content.
- **Contextual Retrieval**: The AI retrieves relevant document chunks to answer queries with accurate references (source and page numbers).
- **Secure API Key Input**: Users can securely enter their **Groq API Key** for authentication.

## Future Additions
- **File Deletion from Database**: Implement functionality to remove uploaded files from the database when no longer needed.
- **Serverless Vector Database - Pinecone**: Integrate Pinecone as an alternative for storing and retrieving embeddings efficiently.
- **Voice Assistant**: Add support for voice-based interactions, allowing users to ask questions and receive verbal responses.

## Installation
### Prerequisites
Ensure you have Python installed (>= 3.8). You also need to set up the environment variables for storing vector embeddings and PDF files.

### Steps
1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-repo.git
   cd your-repo
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Create a `.env` file and define the following variables:
     ```ini
     GROQ_API_KEY=your_api_key_here
     PDF_DIRECTORY=pdfs
     VECTORDB_DIRECTORY=vectordb
     CHROMA_COLLECTION_NAME=your_collection_name
     ```
5. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit interface.
2. Upload one or more PDF files.
3. Wait for the system to process and create the vector store.
4. Enter queries in the chat input to retrieve relevant information.
5. The AI will respond based on extracted document content, citing the **source and page number**.


