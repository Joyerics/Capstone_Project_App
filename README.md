Files:
- app.py
- templates/index.html
- static/styles.css
- ANN_HighUse_Pipeline.pkl
- requirements.txt

Local:
1. python -m venv .venv
2. Windows: .venv\Scripts\activate
3. pip install -r requirements.txt
4. python app.py
5. Open http://127.0.0.1:5000

Render:
1. Replace repo files with this bundle
2. Push to GitHub
3. Manual Deploy -> Deploy latest commit
4. Build command: pip install -r requirements.txt
5. Start command: gunicorn app:app