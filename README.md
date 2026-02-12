Agrimindzz: Kisan Call Centre Query Assistant
An AI-powered agricultural assistant that provides dual-mode responses—Offline (verified records) and Online (generative AI)—to help farmers with crop management, sowing schedules, and pest control.

🚀 Features
Dual-Response System: Compares local, verified KCC records with real-time AI insights.

Offline Mode: Uses FAISS for lightning-fast semantic search across 2,003 agricultural Q&A pairs.

Online Mode: Integrates IBM Watsonx Granite 3.0 (8B) LLM to provide synthesized expert advice.

Context-Aware: The AI identifies if local records are insufficient and provides general expert guidance.

🛠️ Technical Stack
Frontend: Streamlit

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Database: FAISS (Facebook AI Similarity Search)

LLM: IBM Granite-3-8b-instruct

Region: Hosted on IBM Cloud London (eu-gb)
