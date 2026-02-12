import os
import requests
from dotenv import load_dotenv

load_dotenv()

def call_granite_llm(user_query, retrieved_context):
    
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    model_id = os.getenv("MODEL_ID", "ibm/granite-3-8b-instruct")

    if not api_key or not project_id:
        raise ValueError("Missing WATSONX_API_KEY or WATSONX_PROJECT_ID in .env")

    
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    iam_data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    iam_response = requests.post(iam_url, data=iam_data).json()
    iam_token = iam_response["access_token"]

    
    
    context_text = "\n".join([f"- {res['answer']}" for res in retrieved_context])
    
    system_msg = "You are Granite, an AI assistant developed by IBM. You follow instructions and provide concise, expert agricultural answers."
    user_msg = f"Based on these records:\n{context_text}\n\nAnswer the farmer's question: {user_query}"

    
    api_url = "https://eu-gb.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    headers = {
        "Authorization": f"Bearer {iam_token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model_id": model_id,
        "project_id": project_id,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    }

    response = requests.post(api_url, headers=headers, json=payload)
    response_json = response.json()
    
    
    if 'choices' not in response_json:
        print(f"API Response: {response_json}")
        
        
        if 'errors' in response_json:
            error_msg = response_json['errors'][0].get('message', 'Unknown error')
            raise ValueError(f"Watsonx API Error: {error_msg}")
        
        
        if 'results' in response_json:
            return response_json['results'][0]['generated_text'].strip()
        
        raise KeyError(f"Unexpected response format. Keys present: {list(response_json.keys())}")
    
    return response_json['choices'][0]['message']['content'].strip()
