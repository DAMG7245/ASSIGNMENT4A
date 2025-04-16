# ASSIGNMENT4A
# PDF Query Assistant

A Streamlit application that uses multiple Large Language Models (LLMs) to summarize and answer questions about PDF documents.

## Overview

PDF Query Assistant is a web application that allows users to upload PDF documents or select previously parsed PDFs, and then leverages the power of Large Language Models to:
- Generate concise summaries of document content
- Answer specific questions about the document

The application is built with a modern stack including Streamlit for the frontend, FastAPI for the backend, and LiteLLM for managing connections to various LLM providers.

## Project Links and Resources

- **Codelabs Documentation**: [Codelabs Guide](https://codelabs-preview.appspot.com/?file_id=1nJm3Fy18WSof_842AEghsUayO015ErjmJPdaO9sauho/edit?tab=t.5cpih9qtxm58#0)  
- **Project Submission Video (5 Minutes)**: [View on Google Drive](https://drive.google.com/drive/u/1/folders/1898HGutXjQIxwx3OVnr_Yvx9Uq_SKAE1)  
- **Hosted Application Links**:  
  - **Frontend (Streamlit)**: [http://104.248.126.152:8501/](http://104.248.126.152:8501/)  
  - **Backend (FastAPI)**: [http://104.248.126.152:8000/](http://104.248.126.152:8000/)


## Architecture

![pdf_query_assistant_architecture](https://github.com/user-attachments/assets/c0446ca5-f13c-4de8-8695-0281500dbe22)


The system consists of three main components:

1. **Streamlit Frontend**: User interface for uploading PDFs, selecting models, and interacting with the system
2. **FastAPI Backend**: REST API that processes requests and manages communication with LLMs
3. **LiteLLM Integration**: Handles connections to various LLM providers (GPT-4o, Claude, Gemini, etc.)

Communication between components is managed through RESTful API calls and Redis streams.

## Features

- 📄 PDF document upload and parsing
- 🔍 Selection from previously parsed PDF content
- 🤖 Support for multiple LLM providers (OpenAI, Anthropic, Google, DeepSeek, xAI)
- 📝 Document summarization
- ❓ Question answering based on document content
- 📊 Token usage tracking and cost reporting

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **API Management**: LiteLLM
- **Messaging**: Redis Streams
- **Deployment**: Docker Compose
- **Cloud Deployment**: AWS/GCP/Azure (specify your choice)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- API keys for LLM providers of your choice

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-query-assistant.git
cd pdf-query-assistant
```

2. Create a `.env` file with your API keys
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
XAI_API_KEY=your_xai_api_key
```

3. Start the application with Docker Compose
```bash
docker-compose up -d
```

4. Access the application at `http://localhost:8501`

### Manual Setup (Without Docker)

1. Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

2. Install frontend dependencies
```bash
cd frontend
pip install -r requirements.txt
```

3. Start the Redis server
```bash
redis-server
```

4. Start the FastAPI backend
```bash
cd backend
uvicorn main:app --reload
```

5. Start the Streamlit frontend
```bash
cd frontend
streamlit run app.py
```

## Usage

1. Open the Streamlit application in your browser
2. Select an LLM provider from the dropdown
3. Either:
   - Upload a new PDF document
   - Select a previously parsed PDF
4. For summarization:
   - Click the "Summarize" button
5. For question answering:
   - Type your question in the text input
   - Click the "Ask" button
6. View the results in the display area

## API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /select_pdfcontent`: Select previously parsed PDF content
- `POST /upload_pdf`: Upload and process a new PDF document
- `POST /summarize`: Generate a summary of the document
- `POST /ask_question`: Answer a question based on the document content

## Development

### Project Structure

```
pdf-query-assistant/
├── docker-compose.yml
├── docs/
│   └── architecture.png
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── llm_manager.py
│   └── utils/
│       └── pdf_processor.py
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── utils/
│       └── api_client.py
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Monitoring and Logging

The application includes comprehensive logging of:
- API requests and responses
- LLM token usage and costs
- Error tracking

Logs can be viewed through Docker Compose:
```bash
docker-compose logs -f
```

## Deployment

The application is containerized using Docker and can be deployed to any cloud provider that supports Docker containers.

### Cloud Deployment Steps

1. Build the Docker images
```bash
docker-compose build
```

2. Push the images to your container registry
3. Deploy using your cloud provider's container orchestration service

## License

[Specify your license here]

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [OpenAI](https://openai.com/)
- [Anthropic](https://anthropic.com/)
- [Google](https://deepmind.google/)
- [DeepSeek](https://deepseek.ai/)
- [xAI](https://x.ai/)
