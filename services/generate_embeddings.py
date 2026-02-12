import json
import pickle
import os
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_json, output_pickle):
    
    if not os.path.exists(input_json):
        print(f"Error: {input_json} not found. Run Milestone 1 first.")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    
    
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    
    
    print(f"Generating embeddings for {len(qa_data)} entries...")
    texts = [f"Q: {item['question']} A: {item['answer']}" for item in qa_data]
    embeddings = model.encode(texts, show_progress_bar=True)

    
    
    embedded_records = []
    for vec, item in zip(embeddings, qa_data):
        embedded_records.append({
            "embedding": vec,
            "metadata": item
        })

    
    os.makedirs(os.path.dirname(output_pickle), exist_ok=True)

    with open(output_pickle, "wb") as f:
        pickle.dump(embedded_records, f)

    print(f"Success! Embeddings saved to {output_pickle}")

if __name__ == "__main__":
    generate_embeddings(
        input_json="data/kcc_qa_pairs.json",
        output_pickle="embeddings/kcc_embeddings.pkl"
    )
