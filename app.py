from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("ANN_HighUse_Pipeline.pkl")

OPTIONS = {
    "Age": ["15–24 years", "25–34 years", "35–44 years", "45–54 years", "55–64 years", "65 years and over"],
    "Gender": ["Female", "Male"],
    "Education": ["High school or less", "Some post-secondary (incl. certificate)", "University degree"],
    "Employment": ["Employed", "Not employed"],
    "Income_Group": ["≤ $42,256", "$42,257 – $72,366", "$72,367 – $107,480", "$107,481 – $163,750", "≥ $163,751"],
    "Household_Type": ["Single-person household", "Family with children <18", "Family without children <18"],
    "Online_Shopping": ["Online shopper (≥ $1)", "Did not shop online"],
    "Rel_Satisfaction": ["Completely satisfied", "Somewhat satisfied", "Neutral", "Somewhat dissatisfied", "Completely dissatisfied"],
    "SM_Interfere_Rel": ["No", "Yes"],
    "SM_Interfere_Life": ["No", "Yes"],
    "SM_Anxious_Envious": ["No", "Yes"],
    "SM_AI_Chatbot": ["No", "Yes"],
    "SM_AI_Email": ["No", "Yes"]
}

FEATURE_ORDER = [
    "Province","Age","Gender","Education","Employment","Income_Group","Weight",
    "Rel_Satisfaction","SM_Interfere_Rel","SM_Interfere_Life","SM_Anxious_Envious",
    "SM_AI_Chatbot","SM_AI_Email","Household_Type","Online_Shopping"
]

RESEARCH = {
    "screen_time": {
        "title": "Reduce daily screen time",
        "text": "Reducing daily screen time can improve mental health, lower stress, and support better sleep quality.",
        "source_title": "Randomized controlled trial on screen time reduction and mental health",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11846175/"
    },
    "detox": {
        "title": "Take short digital detox breaks",
        "text": "Short breaks from social media can reduce anxiety, depression, and insomnia symptoms.",
        "source_title": "JAMA Network Open study on digital detox",
        "source_url": "https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2841773"
    },
    "lower_usage": {
        "title": "Lower overall social media usage",
        "text": "Reducing overall social media usage is associated with lower depression and better emotional well-being.",
        "source_title": "Meta-analysis on reducing social media use",
        "source_url": "https://www.mdpi.com/2254-9625/15/11/222"
    },
    "sleep_activity": {
        "title": "Protect sleep and physical activity",
        "text": "Better sleep habits and more physical activity can reduce negative effects linked to heavy screen use.",
        "source_title": "Nature study on screen time, sleep, and mental health",
        "source_url": "https://www.nature.com/articles/s41599-026-06609-1"
    },
    "emotional_exposure": {
        "title": "Reduce negative emotional exposure",
        "text": "Limiting stressful, comparison-heavy, or emotionally triggering content can support better mental well-being.",
        "source_title": "JMIR meta-analysis on problematic social media use and mental health",
        "source_url": "https://mental.jmir.org/2022/4/e33450"
    }
}

def build_input(form_data):
    row = {
        "Province": "Ontario",
        "Age": form_data.get("Age"),
        "Gender": form_data.get("Gender"),
        "Education": form_data.get("Education"),
        "Employment": form_data.get("Employment"),
        "Income_Group": form_data.get("Income_Group"),
        "Weight": float(form_data.get("Weight") or 1.0),
        "Rel_Satisfaction": form_data.get("Rel_Satisfaction"),
        "SM_Interfere_Rel": form_data.get("SM_Interfere_Rel"),
        "SM_Interfere_Life": form_data.get("SM_Interfere_Life"),
        "SM_Anxious_Envious": form_data.get("SM_Anxious_Envious"),
        "SM_AI_Chatbot": form_data.get("SM_AI_Chatbot"),
        "SM_AI_Email": form_data.get("SM_AI_Email"),
        "Household_Type": form_data.get("Household_Type"),
        "Online_Shopping": form_data.get("Online_Shopping")
    }
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

def build_recommendations(form_data, probability):
    recs = []
    if probability >= 60:
        recs += [RESEARCH["screen_time"], RESEARCH["detox"], RESEARCH["lower_usage"]]
    recs += [RESEARCH["sleep_activity"]]
    if form_data.get("SM_Anxious_Envious") == "Yes":
        recs += [RESEARCH["emotional_exposure"]]
    out, seen = [], set()
    for r in recs:
        if r["title"] not in seen:
            out.append(r); seen.add(r["title"])
    return out

@app.route("/", methods=["GET", "POST"])
def home():
    submitted = {k: "" for k in OPTIONS}
    weight = "1.0"
    prediction = None
    probability = None
    recommendations = []

    if request.method == "POST":
        submitted = {k: request.form.get(k, "") for k in OPTIONS}
        weight = request.form.get("Weight", "1.0")
        x = build_input(request.form)
        pred = int(model.predict(x)[0])
        prob = float(model.predict_proba(x)[0, 1])
        prediction = "High Social Media Usage" if pred == 1 else "Not-High Social Media Usage"
        probability = round(prob * 100, 1)
        recommendations = build_recommendations(request.form, probability)

    return render_template("index.html", options=OPTIONS, submitted=submitted, weight=weight,
                           prediction=prediction, probability=probability,
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)