# Attention is All You Need – RA Chatbot

This project is a **Streamlit-based chatbot** designed to answer questions about the concepts and content from *“Attention is All You Need”*. It leverages **Retrieval-Augmented Generation (RAG)** using **LangChain**, **OpenAI LLMs**, and **Chroma vector stores**.

---

## Project Overview

The chatbot provides interactive, context-aware answers to questions based on the content of PDFs related to the Transformer paper. Instead of purely generative responses, it uses a **retrieval-based approach**:

1. **Load Documents** – All PDFs in the script’s folder are loaded.  
2. **Split Content** – Documents are split into smaller **token-based chunks** for efficient processing.  
3. **Create Embeddings** – Each chunk is embedded using **OpenAI’s `text-embedding-3-small`** model.  
4. **Vector Storage** – Embeddings are stored in **Chroma** for similarity search.  
5. **RAG Pipeline** – User questions are answered using the most relevant content retrieved from the vector store, passed to **GPT-4o-mini**, and returned as a context-aware response.

---

##  How It Works

### 1. Loading PDFs
- Scans the folder for `.pdf` files.  
- Uses `PyPDFLoader` to extract text from PDFs.  
- Concatenates all document content.

### 2. Splitting Text
- **Token Split:** Divides text into 500-token chunks with 50-token overlap to preserve context.  

### 3. Creating Embeddings
- Embeddings are generated using `OpenAIEmbeddings`.  
- Stored in **Chroma vectorstore** for fast retrieval.

### 4. Retrieval-Augmented Generation (RAG)
- **Retriever:** Uses MMR (Maximal Marginal Relevance) to select relevant chunks.  
- **LLM:** `ChatOpenAI` (GPT-4o-mini) generates answers using retrieved context.  
- **Prompts:** Custom system and human prompts ensure the response is based on the context.

### 5. User Interface
- Built with **Streamlit**.  
- Features:
  - Custom question input.
  - Sample questions sidebar.

---

## Training / Knowledge Base

- Chatbot knowledge is derived from papers on *“Attention is All You Need”*.  
- Text is split into token chunks, embedded, and stored for retrieval.  
- No external data is used; answers are strictly based on the uploaded content.


git clone <repo_url>
cd <repo_folder>
