# ManakSetu: Procurement Intelligence for Indian Standards

**Live Deployment:** [project-manak-setu.vercel.app](https://project-manak-setu.vercel.app)

ManakSetu is an advanced regulatory technology (RegTech) and procurement intelligence platform designed to interface with the Bureau of Indian Standards (BIS) catalogue. By combining hybrid vector search with large language models, the platform analyzes tender documents, technical specifications, and free-text queries to recommend precise compliance requirements, detect regulatory gaps, and map Quality Control Orders (QCOs).

## Core Intelligence Architecture

ManakSetu operates on a foundational Retrieval-Augmented Generation (RAG) architecture utilizing MiniLM embeddings and FAISS/Qdrant vector stores. Rather than relying on model fine-tuning or generative assumptions, the system enforces strict deterministic grounding through six specialized intelligence modules residing in the `ai-engine/src/` package:

1. **Requirement-Level Analysis (`requirement_analysis.py`)**
   Decomposes monolithic tender documents and complex queries into granular, analyzable requirements for precise standard matching.

2. **Semantic Relevance & Ranking (`ranking.py`)**
   Evaluates retrieved standards against specific requirements, scoring them on deep semantic relevance rather than relying solely on raw vector proximity.

3. **Provenance & Evidence Tracking (`evidence.py`)**
   Ensures zero hallucination by grounding every claim directly in the extracted text. Evidence is only emitted from fields explicitly present in the verified record; unverified knowledge-base fields remain strictly flagged as unverified.

4. **Automated Issue Detection (`issue_detector.py`)**
   Scans for outdated designations, conflicting specifications, and QCO compliance failures. High-severity or unverified findings are immediately escalated for human review.

5. **Confidence & Uncertainty Scoring (`confidence.py`)**
   Quantifies the reliability of each match based on retrieval quality and contextual density, openly acknowledging uncertainty when data is thin.

6. **End-to-End Orchestration (`recommender.py`)**
   Synthesizes the outputs of the above modules into a highly structured result payload containing requirements, analyses, recommendations, rankings, evidence, issues, confidence metrics, alerts, and human-review flags.

## Technical Stack

*   **Frontend:** React 18, Vite, Tailwind CSS (Deployed on Vercel)
*   **Backend API:** Python, FastAPI, SQLAlchemy, asyncpg, PyMuPDF (PDF intelligence)
*   **AI Engine:** Python, FastAPI, Google Gemini API, Scikit-Learn (Applicability modeling), FAISS, pgvector
*   **Database:** PostgreSQL

## System Architecture & Deployment

To optimize resource utilization, the backend and AI Engine are designed to run concurrently within a single environment, communicating over local HTTP interfaces. 

### Production Deployment
*   **Frontend:** Hosted independently on Vercel.
*   **Backend & AI Engine:** Orchestrated via `start.sh` to run in a single Render container. 
    *   The AI Engine operates on an internal port (`10001`) in a lightweight inference mode (`SKIP_RECOMMENDER=true`), loading the pre-trained `.joblib` applicability models directly into memory.
    *   The public-facing Backend binds to the environment `$PORT` and routes analysis requests to the internal AI Engine.

## Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kartikey-97/ManakSetu.git
   cd ManakSetu
   ```

2. **Backend & AI Engine Setup:**
   Ensure Python 3.11+ is installed.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt
   pip install -r ai-engine/requirements.txt
   ```
   Set up your `.env` file with required database and API credentials (e.g., `GEMINI_API_KEY`, PostgreSQL URI).

3. **Start the Backend Services:**
   You can utilize the deployment script to launch both the API and the AI engine locally:
   ```bash
   ./start.sh
   ```

4. **Frontend Setup:**
   In a new terminal window:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Disclaimer

ManakSetu generates intelligence based on automated retrieval and semantic analysis. All critical compliance, legal, and procurement decisions flagged for human review must be verified against official BIS publications and government gazettes.
