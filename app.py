import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import numpy as np
import json
import os

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    with open("knowledge-base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    questions = [item["question"] for item in kb]
    question_embeddings = embedding_model.embed_documents(questions)
    question_embeddings = np.array(question_embeddings)
    question_embeddings /= np.linalg.norm(question_embeddings, axis=1, keepdims=True)
    return kb, question_embeddings

@st.cache_data(show_spinner=False)
def get_document_chunks(file_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text_splitter.split_text(text)

@st.cache_resource(show_spinner=False)
def create_faiss_store(chunks):
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        vector_store = FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
        vector_store.save_local(index_dir)
    return vector_store

def main():
    st.title("📚 Simple Embedding Search with FAISS & Knowledge Base")
    st.markdown("Simple embedding store using FAISS to insert document chunks and perform similarity searches over them using vector embeddings")

    # load KB and document chunks
    kb, kb_embeddings = load_knowledge_base()
    chunks = get_document_chunks("genpact-third-party-code-of-conduct.txt")
    st.sidebar.write(f"Loaded {len(chunks)} document chunks.")

    vector_store = create_faiss_store(chunks)

    user_query = st.text_input("Enter your query:", "")

    if user_query:
        query_embedding = embedding_model.embed_query(user_query)
        query_embedding = np.array(query_embedding)
        query_embedding /= np.linalg.norm(query_embedding)

        similarities = np.dot(kb_embeddings, query_embedding.T).flatten()
        best_match_idx = np.argmax(similarities)
        best_match_score = similarities[best_match_idx]

        if best_match_score > 0.85:
            st.success(f"Best KB match (score: {best_match_score:.3f}):")
            st.markdown(f"**Q:** {kb[best_match_idx]['question']}")
            st.markdown(f"**A:** {kb[best_match_idx]['answer']}")
        else:
            st.info("No strong KB match found. Searching document chunks...")

            results = vector_store.similarity_search(user_query, k=3)

            st.markdown("### Top matching document chunks:")
            for i, doc in enumerate(results):
                st.markdown(f"**Result {i + 1}:**")
                st.write(doc.page_content)
                st.markdown("---")

if __name__ == "__main__":
    main()
