# RAG Pipeline Implementation Study

This repository is a comprehensive, step-by-step implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. It is designed as an educational resource for students and developers to understand how to move from raw unstructured data to an intelligent retrieval system.

## The RAG Lifecycle

The project is structured to follow the natural flow of a RAG system:

**Data Ingestion** $\rightarrow$ **Text Splitting** $\rightarrow$ **Vector Storage** $\rightarrow$ **Information Retrieval**

---

## Repository Structure

### 1. `document_loaders/`
Focuses on the **Ingestion** phase. It demonstrates how to handle various data sources:
- **PDF Loader**: Extracting text from academic papers and CVs.
- **CSV Loader**: Handling tabular data.
- **Web Base Loader**: Scraping content from URLs.
- **Directory Loader**: Processing entire folders of documents.

### 2. `text_splitters/`
Focuses on **Chunking**. Since LLMs have limited context windows, this module explores different ways to break text:
- **Length-based**: Simple character or token counts.
- **Structure-based**: Respecting document boundaries (paragraphs, headers).
- **Semantic-based**: Using embeddings to split text where the meaning actually changes.

### 3. `vector_stores/`
Focuses on **Indexing**. This module implements the "memory" of the RAG system:
- Integration with **ChromaDB** to store document embeddings.
- Management of persistence and similarity search.

### 4. `retrievers/`
Focuses on **Context Fetching**. This is where the "intelligence" of retrieval happens:
- **Vector Store Retriever**: Basic similarity search.
- **Wikipedia Retriever**: Fetching external knowledge.
- **Advanced Techniques**: Implementation of **MMR (Maximum Marginal Relevance)** to ensure the retrieved documents are diverse and not redundant.

### 5. `notebooks/`
Contains interactive Jupyter notebooks (`MMR.ipynb`, `MQR.ipynb`, etc.) that provide a visual, step-by-step walkthrough of the retrieval experiments.

---

## Getting Started

### Prerequisites
- Python 3.9+
- An API key for your embedding provider (e.g., OpenAI, HuggingFace)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   cp .evn.example .env
   # Edit .env and add your API keys
   ```

## Learning Path for Students

If you are new to RAG, follow this guided path to understand how the pipeline components connect:

### Step 1: Data Ingestion
**Goal**: Turn a raw file into a string of text.
- Explore the [document_loaders/](./document_loaders/) folder.
- **Key Question**: How does the system handle different formats like PDF vs. CSV?

### Step 2: Text Splitting (Chunking)
**Goal**: Break large text into smaller, meaningful pieces.
- Explore the [text_splitters/](./text_splitters/) folder.
- **Key Question**: Why is "Semantic Splitting" better than just splitting every 500 characters?

### Step 3: Vector Storage (Indexing)
**Goal**: Convert text chunks into vectors and store them for fast search.
- Explore the [vector_stores/](./vector_stores/) folder.
- **Key Question**: How does a Vector Database allow us to search by "meaning" rather than "keywords"?

### Step 4: Intelligent Retrieval
**Goal**: Fetch the most diverse and relevant context for the LLM.
- Explore the [retrievers/](./retrievers/) folder.
- Deep dive into the [notebooks/](./notebooks/) to see **MMR** and **MQR** in action.
- **Key Question**: How does Maximum Marginal Relevance (MMR) prevent the system from giving the LLM three identical paragraphs?
