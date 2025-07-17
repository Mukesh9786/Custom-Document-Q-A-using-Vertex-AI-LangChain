vertex-doc-qa/
│
├── docs/                 # Sample documents
├── app/                  # Main application
│   ├── extractor.py      # PDF/Text extraction logic
│   ├── embedder.py       # Vertex AI embedding logic
│   ├── retriever.py      # Vector DB search
│   ├── generator.py      # Vertex AI PaLM2 LLM response
│   └── app.py            # FastAPI or Streamlit interface
│
├── requirements.txt      # Python dependencies
├── README.md             # Project explanation & setup
└── config.yaml           # GCP project and model config
