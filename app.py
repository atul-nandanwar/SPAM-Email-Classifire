import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# LOAD MODEL
# -----------------------
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return " ".join(words)

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1250px;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef4ff 0%, #f7f9fc 45%, #eef7f3 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 34px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 17px;
    color: #cbd5e1;
    max-width: 700px;
}

.glass-card {
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.55);
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    margin-bottom: 20px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 10px;
}

.section-text {
    color: #475569;
    font-size: 15px;
    line-height: 1.7;
}

.metric-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
    border: 1px solid #e2e8f0;
    text-align: center;
}

.metric-label {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #0f172a;
}

.spam-box {
    background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    border: 1px solid #fecdd3;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: #be123c;
    margin-top: 12px;
}

.ham-box {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 1px solid #a7f3d0;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: #047857;
    margin-top: 12px;
}

.footer-box {
    text-align: center;
    color: #64748b;
    font-size: 14px;
    padding: 12px 0 6px 0;
}

.badge {
    display: inline-block;
    background: #dbeafe;
    color: #1d4ed8;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 14px;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, #2563eb 0%, #0f766e 100%);
    color: white;
    font-size: 17px;
    font-weight: 700;
    box-shadow: 0 12px 24px rgba(37, 99, 235, 0.22);
}

.stButton > button:hover {
    filter: brightness(1.05);
}

textarea {
    border-radius: 14px !important;
}

.sample-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #334155;
    padding: 12px;
    border-radius: 14px;
    margin-bottom: 10px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.markdown("## 🛡️ Spam Shield")
    st.caption("Smart Email Protection")
    st.markdown("---")
    st.markdown("### Model Overview")
    st.markdown("- TF-IDF Vectorizer")
    st.markdown("- Machine Learning Classifier")
    st.markdown("- Real-time Prediction")
    st.markdown("- Confidence Score")

    st.markdown("### Accuracy")
    st.success("98%")

    st.markdown("### About Project")
    st.write(
        "This app detects whether a message is Spam or Not Spam using NLP preprocessing, "
        "TF-IDF feature extraction, and a trained machine learning model."
    )

    st.markdown("### Tech Stack")
    st.write("Python, Streamlit, Scikit-learn, NLTK, Joblib")

# -----------------------
# HERO
# -----------------------
st.markdown("""
<div class="hero-card">
    <div class="badge">Machine Learning Powered Email Security</div>
    <div class="hero-title">Spam Email Classifier</div>
    <div class="hero-subtitle">
        Analyze messages instantly with a clean, modern interface. 
        Paste any email or SMS text and get a fast prediction with confidence score.
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# TOP METRICS
# -----------------------
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">98%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Vectorization</div>
        <div class="metric-value">TF-IDF</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Prediction Type</div>
        <div class="metric-value">Spam / Ham</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------
# MAIN CONTENT
# -----------------------
left, right = st.columns([1.7, 1])

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Paste Email Here</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">Enter any email or message below and click predict to classify it.</div>',
        unsafe_allow_html=True
    )

    email_text = st.text_area(
        "",
        height=240,
        placeholder="Example: Congratulations! You have won a free vacation. Click here to claim now..."
    )

    if st.button("Predict Now"):
        if email_text.strip() == "":
            st.warning("Please enter some message first.")
        else:
            cleaned_text = clean_text(email_text)
            text_vector = vectorizer.transform([cleaned_text])

            prediction = model.predict(text_vector)[0]
            probs = model.predict_proba(text_vector)[0]
            confidence_score = max(probs) * 100

            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown('<div class="spam-box">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ham-box">✅ NOT SPAM</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Confidence", f"{confidence_score:.2f}%")
            with c2:
                st.metric("Predicted Class", "Spam" if prediction == 1 else "Ham")

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-text">
    Text Input<br><br>
    ↓<br><br>
    Preprocessing<br><br>
    ↓<br><br>
    TF-IDF Vectorization<br><br>
    ↓<br><br>
    Model Prediction<br><br>
    ↓<br><br>
    Spam / Not Spam
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sample Messages</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-box">Congratulations! You won a free gift card. Click now.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-box">Hey, are we still meeting tomorrow evening?</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-box">Limited-time loan approval offer. Apply immediately.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-text">A polished NLP mini-project for identifying unwanted or suspicious messages through machine learning.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# FOOTER
# -----------------------
st.markdown(
    '<div class="footer-box">Built with Streamlit • Spam Shield Classifier • Machine Learning Project</div>',
    unsafe_allow_html=True
)