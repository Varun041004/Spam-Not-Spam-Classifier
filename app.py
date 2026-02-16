import streamlit as st
import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title("📩 Spam Message Detector")
st.write("Enter a message to check whether it's Spam or Not Spam")

# -------------------------------
# SIMPLE STOPWORDS LIST
# -------------------------------
stop_words = {
    "a","an","the","is","are","was","were","in","on","at","to","for",
    "of","and","or","but","if","then","this","that","it","as","with",
    "by","from","up","about","into","over","after","below","above",
    "he","she","they","we","you","i","me","him","her","them","my",
    "your","his","their","our","be","been","being","have","has","had",
    "do","does","did","so","because","while","can","will","just","ur"
}

# -------------------------------
# TEXT CLEANING FUNCTION
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------------
# LOAD AND TRAIN MODEL (cached)
# -------------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    data.columns = ["label", "message"]
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data["message"] = data["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=3000
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

model, vectorizer = train_model()

# -------------------------------
# USER INPUT
# -------------------------------
message = st.text_area("Enter your message:")

if st.button("Check Spam"):
    if message:
        msg_clean = clean_text(message)
        msg_vec = vectorizer.transform([msg_clean])

        prob = model.predict_proba(msg_vec)[0][1]

        if model.predict(msg_vec)[0] == 1:
            st.error(f"❌ Spam Message (confidence: {prob:.2f})")
        else:
            st.success(f"✅ Not Spam (confidence: {1-prob:.2f})")
    else:
        st.warning("Please enter a message.")
