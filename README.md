# 🚗 Insurance Claims Support AI Agent with LangMem & RAG

An AI-powered Insurance Claims Support Copilot that assists human support agents and claims adjusters in handling **First Notice of Loss (FNOL)** workflows. The system combines **Retrieval-Augmented Generation (RAG)**, **LangMem long-term memory**, **tool calling**, and **human-in-the-loop approval** to generate grounded, context-aware claim recommendations.

---

## 📌 Overview

This project simulates a production-grade insurance claims support system where AI assists—not replaces—human claims adjusters.

The copilot:

- Retrieves relevant insurance policy documents using RAG
- Searches long-term customer and company memory using LangMem
- Invokes structured operational tools
- Generates editable claim recommendations using Groq LLM
- Stores approved resolutions as reusable memories for future claims

Final decisions always remain with the human reviewer.

---

## ✨ Features

- 📄 Insurance Knowledge Base using ChromaDB
- 🧠 Long-Term Memory using LangMem
- 🤖 AI-powered Draft Recommendation Generation
- 🔍 Retrieval-Augmented Generation (RAG)
- 🛠️ Tool Calling for operational information
- 👨‍💼 Human-in-the-loop approval workflow
- 📊 Interactive Streamlit Dashboard
- ⚡ FastAPI REST Backend
- 🗄️ SQLite Database
- 🐳 Docker & Docker Compose Support
- ✅ GitHub Actions & Pytest Integration

---

# 🏗️ System Architecture

```
                    Human Support Agent
                             │
                             ▼
                   Streamlit Dashboard
                             │
                             ▼
                      FastAPI Backend
                             │
        ┌────────────┬───────────────┬─────────────┐
        │            │               │             │
        ▼            ▼               ▼             ▼
   LangMem      ChromaDB RAG     Support Tools   SQLite
    Memory       Knowledge Base                 Database
        │            │               │
        └────────────┴───────────────┘
                     │
                     ▼
              Groq LLM Generation
                     │
                     ▼
           Draft Recommendation
                     │
          Human Review & Approval
                     │
                     ▼
        Accepted Draft → Stored as Memory
```

---

# 📂 Project Structure

```
.
├── customer_support_agent/
│   ├── api/
│   ├── core/
│   ├── integrations/
│   │   ├── memory/
│   │   ├── rag/
│   │   └── tools/
│   ├── repositories/
│   ├── routers/
│   ├── schemas/
│   └── services/
│
├── knowledge_base/
├── data/
├── app.py
├── main.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# 🧠 AI Pipeline

1. User registers an insurance claim
2. Customer and company memories are retrieved using LangMem
3. Relevant insurance policy documents are fetched using RAG
4. Operational tools provide structured information
5. Groq LLM generates an AI recommendation
6. Human reviews and edits the draft
7. Approved draft is stored as reusable memory

---

# 🛠️ Tech Stack

## Backend

- FastAPI
- Python 3.11
- Uvicorn
- SQLite
- Pydantic

## AI

- LangChain
- LangGraph
- LangMem
- Groq (Llama 3.1)
- Google Gemini Embeddings

## Retrieval

- ChromaDB
- RecursiveCharacterTextSplitter

## Frontend

- Streamlit

## DevOps

- Docker
- Docker Compose
- GitHub Actions
- uv

---

# 📊 Database Schema

The application uses SQLite with three primary tables:

- Customers
- Tickets
- Drafts

---

# 📚 Knowledge Base

Insurance documents are stored as Markdown files inside:

```
knowledge_base/
```

Example documents include:

- FNOL Intake Checklist
- Coverage & Deductible Guidelines
- Required Claim Documents
- Settlement SLA
- Fraud Risk Indicators

These documents are chunked and indexed into ChromaDB for semantic retrieval.

---

# 🧠 LangMem Memory

The memory layer stores information at two levels:

- Customer Memory
- Company Memory

Approved claim resolutions become reusable memories, enabling future recommendations to become increasingly context-aware.

---

# 🔧 Tool Calling

The AI agent invokes structured tools during draft generation, including:

- Customer Plan Lookup
- Open Ticket Load Lookup

This enriches recommendations with operational business context.

---

# 👨‍💼 Human-in-the-Loop Workflow

The AI never makes autonomous insurance decisions.

Support agents can:

- Generate recommendations
- Edit drafts
- Approve or discard responses
- Save approved resolutions into memory

This ensures compliance and accountability in insurance workflows.

---

# 📸 Dashboard Features

The Streamlit dashboard includes:

- Claim Registration
- Claim Listing
- Draft Generation
- Draft Editing
- Approval / Reject Workflow
- Knowledge Base Ingestion
- Memory Inspection
- Context Visualization

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Insurance-Claims-Support-AI-Agent.git
cd Insurance-Claims-Support-AI-Agent
```

Install dependencies:

```bash
uv sync
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
```

---

# ▶️ Running the Backend

```bash
uv run main.py
```

Backend:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```

---

# ▶️ Running the Dashboard

```bash
streamlit run app.py
```

Dashboard:

```
http://localhost:8501
```

---

# 🐳 Docker

Build and start the application:

```bash
docker compose up --build
```

---

# 📈 Learning Outcomes

This project demonstrates:

- Production-grade FastAPI architecture
- Retrieval-Augmented Generation (RAG)
- LangMem long-term memory integration
- Tool-calling AI agents
- Human-in-the-loop AI workflows
- Vector databases with ChromaDB
- Dockerized deployment
- Streamlit dashboard development
- Modular software engineering practices

---

# 🚀 Future Improvements

- Multi-user authentication
- Role-based access control
- PostgreSQL support
- Redis caching
- Async task queue
- PDF claim document processing
- Email integration
- OCR for insurance documents
- Kubernetes deployment

---

# 👨‍💻 Author

**Akhil Tiwari**

- GitHub: https://github.com/Akhil6e
- LinkedIn: https://linkedin.com/in/<your-linkedin>

---

## ⭐ If you found this project useful, consider giving it a Star!
