import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from models.granite_llm import call_granite_llm

# Page configuration
st.set_page_config(page_title="KCC Query Assistant", layout="centered")

# Initialize models and data
@st.cache_resource
def load_resources():
    # Load the same model used in Milestone 2
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load the FAISS index and metadata from Milestone 3
    index = faiss.read_index("vector_store/faiss.index")
    with open("vector_store/meta.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata

embedding_model, faiss_index, metadata = load_resources()

# UI Header
st.title("🌾 Kisan Call Centre Query Assistant")
st.markdown("Ask agricultural questions and get verified answers from KCC records and IBM Granite AI.")

# User input
query = st.text_input("Enter your agricultural query:", placeholder="e.g., When to sow moong?")

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Searching records and generating AI response..."):
            # Step 1: Semantic Search (Offline)
            query_vec = embedding_model.encode([query]).astype("float32")
            # Retrieve top-k (e.g., 3) results
            distances, indices = faiss_index.search(query_vec, k=3)
            
            # Extract results from metadata
            retrieved_results = [metadata[idx] for idx in indices[0]]
            
            # Step 2: LLM Synthesis (Online)
            # Pass the results as context to your Granite model
            ai_answer = call_granite_llm(query, retrieved_results)
            
            # Step 3: Display
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📚 Offline (KCC Records)")
                for i, res in enumerate(retrieved_results, 1):
                    st.info(f"**Record {i}:** {res['answer']}")
            
            with col2:
                st.subheader("🤖 Online (IBM Granite)")
                st.success(ai_answer)
    else:
        st.warning("Please enter a question first.")