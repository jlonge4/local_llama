# Local Llama

This project enables you to chat with your PDFs, TXT files, or Docx files entirely offline, free from OpenAI dependencies. It's an evolution of the gpt_chatwithPDF project, now leveraging local LLMs for enhanced privacy and offline functionality.

## Features

- Offline operation: Run in airplane mode
- Local LLM integration: Uses Ollama for improved performance
- Multiple file format support: PDF, TXT, DOCX, MD
- Persistent vector database: Reusable indexed documents
- Streamlit-based user interface

## New Updates

- Ollama integration for significant performance improvements
- Uses nomic-embed-text and llama3:8b models (can be changed to your liking)
- Upgraded to Haystack 2.0
- Persistent Chroma vector database to enable re-use of previously updloaded docs

## Installation

1. Install Ollama from https://ollama.ai/download
2. Clone this repository
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Pull required Ollama models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3:8b
   ```

## Usage

1. Start the Ollama server:
   ```
   ollama serve
   ```
2. Run the Streamlit app:
   ```
   python -m streamlit run local_llama_v3.py
   ```
3. Upload your documents and start chatting!

## How It Works

1. Document Indexing: Uploaded files are processed, split, and embedded using Ollama.
2. Vector Storage: Embeddings are stored in a local Chroma vector database.
3. Query Processing: User queries are embedded and relevant document chunks are retrieved.
4. Response Generation: Ollama generates responses based on the retrieved context and chat history.


## License

This project is licensed under the Apache 2.0 License.

## Acknowledgements

- Ollama team for their excellent local LLM solution
- Haystack for providing the RAG framework
- The-Bloke for the GGUF models
