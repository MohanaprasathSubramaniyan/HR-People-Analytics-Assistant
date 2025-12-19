# 🏢 HR & People Analytics Assistant

**A Secure, Local AI Dashboard for HR Policy Insights and Workforce Data Analysis.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.32.0-ff4b4b)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Llama 3](https://img.shields.io/badge/Model-Llama%203-purple)

---

## 📖 Project Overview

The **HR & People Analytics Assistant** is an offline, privacy-first AI application designed to help HR teams and managers interact with their data using natural language. It eliminates the need for complex SQL queries or manual document searching by combining **RAG (Retrieval-Augmented Generation)** for documents and **Agentic AI** for structured data.

### 🌟 Key Features
* **📄 HR Policy Assistant (RAG Pipeline):**
    * Ingests PDF documents (Employee Handbooks, Code of Conduct, etc.).
    * Uses vector search (ChromaDB) to retrieve relevant clauses.
    * Provides evidence-based answers with source citations.
* **📊 Workforce Analytics Engine (Agentic Pipeline):**
    * Connects to employee CSV datasets.
    * Translates natural language questions (e.g., *"What is the average salary by department?"*) into executable Python/Pandas code.
    * Visualizes data insights instantly.
* **🔒 100% Local & Secure:**
    * Powered by **Ollama (LLaMA 3)** running locally.
    * No data is sent to external cloud providers (OpenAI/Google), ensuring complete data privacy for sensitive employee records.

---

## 🛠️ System Architecture

The application is built on a dual-pipeline architecture:

1.  **Unstructured Data Pipeline:** `PDFs` $\rightarrow$ `Recursive Character Splitter` $\rightarrow$ `HuggingFace Embeddings` $\rightarrow$ `ChromaDB` $\rightarrow$ `LLM Generation`.
2.  **Structured Data Pipeline:** `CSV` $\rightarrow$ `LangChain Pandas Agent` $\rightarrow$ `Python REPL` $\rightarrow$ `Statistical Output`.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following installed on your machine:
* **Python 3.9+**
* **[Ollama](https://ollama.com/)** (for running the local LLM)
* **Git**

### 2. Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/hr-analytics-assistant.git](https://github.com/yourusername/hr-analytics-assistant.git)
    cd hr-analytics-assistant
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the LLaMA 3 Model**
    Open your terminal/command prompt and run:
    ```bash
    ollama pull llama3
    ```

### 3. Data Setup

1.  **PDF Documents:** Place your HR policy PDFs (e.g., `handbook.pdf`) inside the project folder.
2.  **Employee Data:** Ensure your `employees.csv` file is in the root directory.
    * *Required Columns:* `Name`, `Department`, `Position`, `Salary` (or similar).
3.  **Build the Vector Database:**
    Run the ingestion script to process the PDFs:
    ```bash
    python ingest_docs.py
    ```
    *(This creates the `chroma_db` folder).*

---

## 🖥️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
