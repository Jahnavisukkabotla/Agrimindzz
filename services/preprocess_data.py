import pandas as pd
import json
import os

def clean_kcc_data(input_path, output_json):
    print(f"Loading raw data from: {input_path}")
    
    # Load raw CSV (Note: Some KCC files use specific encodings or quote characters)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 1. Selection: Keep only columns needed for the AI Knowledge Base
    # Note: Column names must match your CSV exactly (e.g., 'QueryText', 'KccAns')
    required_columns = ['QueryText', 'KccAns', 'StateName', 'DistrictName', 'Crop']
    df = df[required_columns]

    # 2. Cleaning: Remove empty entries and nonsensical responses
    df = df.dropna(subset=['QueryText', 'KccAns'])
    
    # Optional: Filter out very short or symbol-only answers
    df = df[df['KccAns'].str.len() > 5] 

    # 3. Deduplication: Remove redundant farmer queries
    df = df.drop_duplicates(subset=['QueryText'])

    # 4. Normalization: Strip extra spaces
    df['QueryText'] = df['QueryText'].str.strip()
    df['KccAns'] = df['KccAns'].str.strip()

    # 5. Export to AI-Ready JSON Format
    qa_pairs = []
    for _, row in df.iterrows():
        qa_pairs.append({
            "question": row['QueryText'],
            "answer": row['KccAns'],
            "metadata": {
                "state": row['StateName'],
                "district": row['DistrictName'],
                "crop": row['Crop']
            }
        })

    # Save the output file in your project data directory
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"Milestone 1 Complete: {len(df)} cleaned Q&A pairs saved to {output_json}")

if __name__ == "__main__":
    # Ensure paths align with your kcc_project directory structure
    clean_kcc_data(
        input_path='data/raw_kcc.csv',
        output_json='data/kcc_qa_pairs.json'
    )