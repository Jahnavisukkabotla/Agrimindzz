import faiss
import pickle
import numpy as np
import os

def build_faiss_index(embeddings_file, index_output, meta_output):
    # 1. Load the embeddings generated in Milestone 2
    if not os.path.exists(embeddings_file):
        print(f"Error: {embeddings_file} not found. Run Milestone 2 first.")
        return

    with open(embeddings_file, "rb") as f:
        records = pickle.load(f)

    # 2. Prepare data for FAISS
    # FAISS requires a numpy array of type float32
    embeddings = np.array([r["embedding"] for r in records]).astype("float32")
    metadata = [r["metadata"] for r in records]

    # 3. Create and populate the index
    dimension = embeddings.shape[1] # Should be 384
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 4. Save the index and the metadata separately
    # Metadata is needed because the index only returns ID numbers
    os.makedirs(os.path.dirname(index_output), exist_ok=True)
    
    faiss.write_index(index, index_output)
    with open(meta_output, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Success! Indexed {index.ntotal} vectors.")
    print(f"Index saved to {index_output}")
    print(f"Metadata saved to {meta_output}")

if __name__ == "__main__":
    build_faiss_index(
        embeddings_file="embeddings/kcc_embeddings.pkl",
        index_output="vector_store/faiss.index",
        meta_output="vector_store/meta.pkl"
    )