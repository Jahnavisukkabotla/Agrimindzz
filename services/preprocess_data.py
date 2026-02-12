import pandas as pd
import json
import os

def clean_kcc_data(input_path, output_json):
    print(f"Loading raw data from: {input_path}")
    
    
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    
    
    required_columns = ['QueryText', 'KccAns', 'StateName', 'DistrictName', 'Crop']
    df = df[required_columns]

    
    df = df.dropna(subset=['QueryText', 'KccAns'])
    
    
    df = df[df['KccAns'].str.len() > 5] 

    
    df = df.drop_duplicates(subset=['QueryText'])

    
    df['QueryText'] = df['QueryText'].str.strip()
    df['KccAns'] = df['KccAns'].str.strip()

    
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

    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)

    print(f"Milestone 1 Complete: {len(df)} cleaned Q&A pairs saved to {output_json}")

if __name__ == "__main__":
    
    clean_kcc_data(
        input_path='data/raw_kcc.csv',
        output_json='data/kcc_qa_pairs.json'
    )
