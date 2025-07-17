# Custom-Document-Q-A-using-Vertex-AI-LangChain
gen ai 1

"End-to-End Custom Document Q&A System using Vertex AI + LangChain"

🚀 Project Description:
Build an intelligent Question-Answering system that takes PDF documents (or any text files), embeds them using Vertex AI Embedding Models, stores them in a Vector Database (e.g., FAISS), and allows users to ask questions. The system will respond with accurate answers using Vertex AI PaLM 2 LLM via LangChain or native API integration.

🧠 Key Features:
Upload documents (PDF, DOCX, TXT)

Extract & chunk text from documents

Embed using Vertex AI Text Embedding Model

Store embeddings in FAISS or ChromaDB

Use LangChain or custom Python code to retrieve relevant chunks

Query using Vertex AI PaLM 2 model for answering

Streamlined, clean Flask or FastAPI interface (optional)

Deploy on GCP App Engine or Cloud Run (optional)

🧰 Tech Stack:
Vertex AI (Text Embedding & Text Generation)

Python (LangChain or native client)

FAISS or ChromaDB (for vector search)

Streamlit / FastAPI / Flask (optional UI)

GCP services: Cloud Storage, Vertex AI, Cloud Functions (optional)

GitHub Actions for CI/CD (optional)


✅ Project initialized: Your full Vertex AI-based Document Q&A project code is ready in the file main.py.

📌 Next Steps to Set Up Locally:
Install required packages:

bash
Copy
Edit
pip install google-cloud-aiplatform langchain PyPDF2 faiss-cpu
Authenticate with Google Cloud:

bash
Copy
Edit
gcloud auth application-default login
gcloud config set project your-gcp-project-id
Replace in code:

python
Copy
Edit
PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
Add a sample PDF as sample.pdf in your directory.

Run the app:

bash
Copy
Edit
python main.py
You’ll see prompts like:

css
Copy
Edit
Ask a question:
Ask something about your uploaded document (e.g., "What is the main topic of this paper?").

