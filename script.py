from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import numpy as np
import json

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

def load_knowledge_base():
    with open("knowledge-base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    questions = [item["question"] for item in kb]

    # Use embedding_model to get embeddings
    question_embeddings = embedding_model.embed_documents(questions)
    # Normalize embeddings for cosine similarity
    question_embeddings = np.array(question_embeddings)
    question_embeddings /= np.linalg.norm(question_embeddings, axis=1, keepdims=True)
    return kb, question_embeddings

def get_document_chunks(file_path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text_splitter.split_text(text)

def create_faiss_store(chunks):
    vector_store = FAISS.from_texts(chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    print("FAISS index saved successfully.")

def perform_similarity_search(query, top_k=3):
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    results = vector_store.similarity_search(query, k=top_k)
    return results

def main():
    file_path = "genpact-third-party-code-of-conduct.txt"
    kb, kb_embeddings = load_knowledge_base()
    chunks = get_document_chunks(file_path)
    print(f"✅ Total document chunks created: {len(chunks)}")

    create_faiss_store(chunks)

    while True:
        user_query = input("\nEnter your query (or type 'exit' to stop): ")
        if user_query.lower() == "exit":
            print("👋 Exiting the search.")
            break

        # Embed user query and normalize
        query_embedding = embedding_model.embed_query(user_query)
        query_embedding = np.array(query_embedding)
        query_embedding /= np.linalg.norm(query_embedding)

        # Compute cosine similarity with KB
        similarities = np.dot(kb_embeddings, query_embedding.T).flatten()
        best_match_idx = np.argmax(similarities)
        best_match_score = similarities[best_match_idx]

        if best_match_score > 0.7:
            print(f"\n✅ Best KB match (score: {best_match_score:.3f}):")
            print(f"Q: {kb[best_match_idx]['question']}")
            print(f"A: {kb[best_match_idx]['answer']}")
        else:
            print("\nNo strong KB match found. Searching document chunks...")

        results = perform_similarity_search(user_query)
        print("\nTop matching document results:")
        for i, doc in enumerate(results):
            print(f"\nResult {i + 1}:\n{'-' * 30}")
            print(doc.page_content)

if __name__ == "__main__":
    main()