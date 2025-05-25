# Simple Embedding Search App
This Streamlit app creates a simple embedding store using FAISS to insert document chunks and perform similarity searches using vector embeddings. It allows you to query over an HR Policy document of Genpact to retrieve the most relevant matches.

## 🚀 Features
✅ Ask a query regarding Third Party Code of Conduct Policy for Genpact
✅ Generate embeddings using **`BAAI/bge-base-en-v1.5`** (via Hugging Face)  
✅ Store and search embeddings efficiently using FAISS  
✅ Compare user queries against a predefined knowledge base  
✅ Streamlit web interface for easy interaction

## 📦 Setup

1️⃣ **Clone the repository:**
```
git clone https://github.com/paridhi3/AI-POC.git
cd AI-POC

2️⃣ **Create a virtual environment:**
```
conda create -p venv
conda activate venv/
```

3️⃣ **Install dependencies:**
```
pip install -r requirements.txt
```

▶️ **Run the app locally:**
```
streamlit run app.py
```