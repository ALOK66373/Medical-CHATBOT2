# Medical Chatbot

A Flask-based medical Q&A chatbot that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions from medical documents.

## Features

- Conversational chat interface
- PDF-based document retrieval
- LangChain + Chroma vector search
- Google Gemini model integration
- Docker support for deployment

## Project Structure

- `app.py` - Main Flask application
- `src/helper.py` - Document loading, preprocessing, and embedding utilities
- `src/prompt.py` - Prompt template for the chatbot
- `templates/chat.html` - Web UI for the chat interface
- `static/style.css` - Styling for the chat page
- `data/` - PDF documents used for retrieval
- `chroma_db/` - Persisted vector database

## Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)
- Google API key for Gemini
- Environment variables for the app

## Environment Variables

Create a `.env` file in the project root with:

```env
SECRET_KEY=your-secret-key
GOOGLE_API_KEY=your-google-gemini-api-key
```

Optional deployment variables:

```env
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=your-region
PINECONE_API_KEY=your-pinecone-key
```

## Local Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

The app will be available at:

```text
http://localhost:8080
```

## Docker Setup

Build the image:

```bash
docker build -t medical-chatbot .
```

Run the container:

```bash
docker run -p 8080:8080 \
  -e SECRET_KEY=your-secret-key \
  -e GOOGLE_API_KEY=your-google-gemini-api-key \
  medical-chatbot
```

## How It Works

1. PDFs from the `data/` folder are loaded.
2. Documents are split into chunks.
3. Embeddings are generated and stored in Chroma.
4. The chatbot retrieves relevant chunks and uses Gemini to generate an answer.

## Notes

- The first run may take time because embeddings and the vector index are being generated.
- Make sure your API keys are valid before running the app.

## License

This project is for educational and demonstration purposes.
