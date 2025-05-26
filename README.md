# Simple Embedding Search App
[Link to App](https://paridhi3-ai-poc-app-2c7mfi.streamlit.app/)<br>
This Streamlit app creates a simple embedding store using FAISS to insert document chunks and perform similarity searches using vector embeddings. It allows you to query over an [HR Policy document of Genpact](https://genpact.gcs-web.com/static-files/6a40ce74-dbcc-44d5-88e6-885fe852dade) to retrieve the most relevant matches.

The textual data from the aforementiond PDF was extracted and put into [this file](genpact-third-party-code-of-conduct.txt). From this, a JSON knowledge base was created containing 50 potential FAQs, ensuring that only pure text was extracted while ignoring images and other non-textual content. This knowledge base was then used to complete the task.

## ▶️ Demo
<div align="center">
<video src="https://private-user-images.githubusercontent.com/75247327/447434953-78eb26ef-b613-4f74-9adc-f4c9d218485f.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDgyNDAyNDAsIm5iZiI6MTc0ODIzOTk0MCwicGF0aCI6Ii83NTI0NzMyNy80NDc0MzQ5NTMtNzhlYjI2ZWYtYjYxMy00Zjc0LTlhZGMtZjRjOWQyMTg0ODVmLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA1MjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNTI2VDA2MTIyMFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZiYmIwNjgyNGZjOWEzN2IyODE0YTJjZTVmZWZjOTQxYmQ0NDY4Nzk0MzNiMjkyN2I2ODM0ZWNkNGY1OWQzMTgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.T6nOOZXOJyIBwV6I9YZ5b4X-GI3RDLdGIi2BRY0zbTo" alt="Demo video" width="90%" style="border-radius: 16px;"></video>
</div>

## 🚀 Features
✅ Ask a query regarding Third-Party Code of Conduct Policy for Genpact  
✅ Generate embeddings using **`BAAI/bge-base-en-v1.5`** (via Hugging Face)  
✅ Store and search embeddings efficiently using FAISS  
✅ Compare user queries against a predefined knowledge base  
✅ Streamlit web interface for easy interaction

# 📚 How this works

1️⃣ **Start Streamlit app**  

2️⃣ **Load data**
- Load **knowledge base** (from `knowledge-base.json`) and generate question embeddings.
- Load and split **input document** (from [genpact-third-party-code-of-conduct.txt](genpact-third-party-code-of-conduct.txt)) into small overlapping chunks.
- Create or load **FAISS vector store** with embeddings of document chunks.

3️⃣ **Wait for user input**
- User types a **query** in the input box.

4️⃣ **Process the query**
- Convert the user query into an embedding and normalize it.

5️⃣ **Search knowledge base (KB)**
- Calculate similarity scores between the query and all KB questions.
- Identify the **best-matching question** and its similarity score.

6️⃣ **Check KB match quality**
- **If best KB match score > 0.85** →  
 ✅ Show the KB’s question + answer to the user.
- **Else (no strong KB match)** →  
 ❌ Proceed to search in document chunks.

7️⃣ **Search document chunks (FAISS)**
- Use FAISS to perform **top-3 similarity search** over document chunks.
- Display the top matching chunks to the user.

8️⃣ **Display results**
- Show either:
    - **KB match (Q&A)**  
    **OR**  
    - **Top relevant document chunks** with content.

9️⃣ **End / wait for next query**

---

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