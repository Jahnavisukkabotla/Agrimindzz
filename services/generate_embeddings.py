import json
import pickle
import os
from sentence_transformers import SentenceTransformer

def generate_embeddings(input_json, output_pickle):
    # 1. Load the cleaned Q&A data
    if not os.path.exists(input_json):
        print(f"Error: {input_json} not found. Run Milestone 1 first.")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # 2. Initialize the SentenceTransformer model
    # This model is lightweight and runs fast on CPU
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Prepare texts and generate embeddings
    # We combine Q and A to give the vector more context
    print(f"Generating embeddings for {len(qa_data)} entries...")
    texts = [f"Q: {item['question']} A: {item['answer']}" for item in qa_data]
    embeddings = model.encode(texts, show_progress_bar=True)

    # 4. Save as a list of dictionaries (embedding + metadata)
    # This allows us to retrieve the text later in Milestone 3
    embedded_records = []
    for vec, item in zip(embeddings, qa_data):
        embedded_records.append({
            "embedding": vec,
            "metadata": item
        })

    # Ensure the embeddings directory exists
    os.makedirs(os.path.dirname(output_pickle), exist_ok=True)

    with open(output_pickle, "wb") as f:
        pickle.dump(embedded_records, f)

    print(f"Success! Embeddings saved to {output_pickle}")

if __name__ == "__main__":
    generate_embeddings(
        input_json="data/kcc_qa_pairs.json",
        output_pickle="embeddings/kcc_embeddings.pkl"
    )