# Simple Embedding Search App
[Link to App](https://paridhi3-ai-poc-app-2c7mfi.streamlit.app/)<br>
This Streamlit app creates a simple embedding store using FAISS to insert document chunks and perform similarity searches using vector embeddings. It allows you to query over an [HR Policy document of Genpact](https://genpact.gcs-web.com/static-files/6a40ce74-dbcc-44d5-88e6-885fe852dade) to retrieve the most relevant matches.

The textual data from the aforementiond PDF was extracted and put into [this file](genpact-third-party-code-of-conduct.txt). From this, a JSON knowledge base was created containing 50 potential FAQs, ensuring that only pure text was extracted while ignoring images and other non-textual content. This knowledge base was then used to complete the task.

## 🚀 Features
✅ Ask a query regarding Third-Party Code of Conduct Policy for Genpact  
✅ Generate embeddings using **`BAAI/bge-base-en-v1.5`** (via Hugging Face)  
✅ Store and search embeddings efficiently using FAISS  
✅ Compare user queries against a predefined knowledge base  
✅ Streamlit web interface for easy interaction

## 📦 Setup

1️⃣ **Clone the repository:**
```
git clone https://github.com/paridhi3/AI-POC.git
cd AI-POC
```

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