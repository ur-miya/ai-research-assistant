# AI Research Assistant

An intelligent research assistant that helps you find and analyze academic papers from arXiv using semantic search, metadata filtering, and LLM-powered analysis. Built with LangGraph, ChromaDB, and Hugging Face.

## Features

- **Semantic search** over arXiv papers (up to 5000 samples) using sentence embeddings.
- **Filter by relevant categories** (Machine Learning, Computer Vision, NLP, etc.).
- **Multilingual support**: accepts queries in Russian or English; automatically translates Russian queries to English for better search results.
- **AI-generated answers**: uses Hugging Face Inference API (`mistralai/Mistral-7B-Instruct-v0.2`) to provide structured, insightful responses based on retrieved papers.
- **Step‑by‑step logging**: shows the agent's reasoning process (planning, search, filtering, analysis, answer generation).
- **Interactive command‑line interface** for easy testing and demonstration.

## Tech Stack

- **Python 3.10+**
- **LangGraph** – orchestrates the agent as a stateful graph.
- **ChromaDB** – local vector database for storing and querying paper embeddings.
- **Sentence Transformers** (`all-MiniLM-L6-v2`) – creates embeddings for papers and queries.
- **Hugging Face Inference API** – powers translation and answer generation (model: `mistralai/Mistral-7B-Instruct-v0.2`).
- **LangChain Hugging Face integration** – used in initial versions; final version uses direct `huggingface-hub` client.
- **python-dotenv** – manages the Hugging Face API token.

## Prerequisites

- Python 3.10 or higher
- A [Hugging Face account](https://huggingface.co/) and an API token (free tier available)
- ~2 GB of free disk space (for the vector DB and sample data)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

We provide a pinned `requirements.txt` to avoid dependency conflicts (tested together):

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`** (you can copy this into a file):

```txt
langchain-core>=0.1.7,<0.2.0
langchain==0.1.0
sentence-transformers>=2.7.0,<3.0.0
transformers>=4.42.0,<5.0.0
tokenizers>=0.19.1,<0.21.0
huggingface-hub>=0.26.0,<1.0
langchain-huggingface==0.0.2
langgraph==0.0.30
chromadb==0.4.22
python-dotenv==1.0.0
tqdm==4.66.2
```

### 4. Set up environment variables

Create a `.env` file in the project root and add your Hugging Face API token:

```
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

Replace `hf_xxxxxxxxxxxxxxxxxxxxx` with your actual token (obtain it from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

## Preparing the Data

### Download a sample of arXiv papers

Place your `arxiv_sample.json` file (in JSON Lines format) in the project folder. If you don't have one, you can download a sample from [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) (use the first 5000 lines).

### Run the data preparation script

```bash
python prepare_data.py
```

This will:
- Load up to 5000 articles from `arxiv_sample.json`.
- Create a unified text (`Title: ...\n\nAbstract: ...`) for each paper.
- Save the processed documents as `processed_docs.json`.

### Index the documents into ChromaDB

```bash
python index_to_chromadb.py
```

This script:
- Creates a persistent ChromaDB collection named `arxiv_papers`.
- Generates embeddings for each paper using `all-MiniLM-L6-v2`.
- Stores embeddings, metadata (title, authors, categories, date), and a short preview of the abstract.
- Performs a test search to verify everything works.

You only need to run indexing once. The database is saved locally in the `chromadb_data` folder.

## Running the Agent

Start the interactive agent:

```bash
python research_agent_with_filter.py
```

You will see a prompt. Enter your query (in English or Russian) and the agent will:

1. **Plan**: If the query contains Russian, it translates it to English using Hugging Face.
2. **Search**: Retrieves up to 30 papers from ChromaDB based on semantic similarity.
3. **Filter**: Keeps only papers from relevant arXiv categories (Machine Learning, AI, Computer Vision, etc.) and sorts them by a combination of category priority and relevance score.
4. **Analyze**: Generates a brief analysis of the top papers (distribution by category, top‑3 papers with abstracts).
5. **Answer**: Calls Hugging Face again to produce a well‑structured answer in the user's language, summarising the key findings.

Type `quit` to exit.

### Example queries

```
neural networks transformers
knowledge distillation BERT
computer vision CNN
последние достижения в компьютерном зрении
recent advances in computer vision
```

## Project Structure

```
ai-research-assistant/
├── .env                          # Hugging Face API token
├── .gitignore
├── prepare_data.py                # Load and clean raw arXiv data
├── index_to_chromadb.py           # Create embeddings and index into ChromaDB
├── research_agent_with_filter.py  # Main agent code (LangGraph + Hugging Face)
├── requirements.txt               # Pinned dependencies
├── processed_docs.json            # Output of prepare_data.py (created after step 1)
├── chromadb_data/                 # Persistent ChromaDB storage (created after step 2)
└── README.md                      # This file
```
