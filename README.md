# Clean Premium Flask App

Files:
- app.py
- templates/index.html
- static/styles.css
- ANN_HighUse_Pipeline.pkl
- requirements.txt

Local run:
1. Open terminal in this folder
2. python -m venv .venv
3. Windows: .venv\Scripts\activate
4. pip install -r requirements.txt
5. python app.py
6. Open http://127.0.0.1:5000

Render deploy:
1. Upload all files in this folder to GitHub
2. Create New Web Service in Render
3. Build command: pip install -r requirements.txt
4. Start command: gunicorn app:app
5. Deploy
