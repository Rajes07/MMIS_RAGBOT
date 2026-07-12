python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env          # then add your real GROQ_API_KEY
# drop PDFs into data/pdfs/, matching topics.json filenames
python ingest.py
python query.py               # optional CLI test before the UI
streamlit run app.py