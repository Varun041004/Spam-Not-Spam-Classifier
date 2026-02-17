import streamlit as st
import pickle
import string
import re

st.title("📩 Spam Message Detector")
st.write("Enter a message to check whether it's Spam or Not Spam")

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

stop_words = {
    "a","an","the","is","are","was","were","in","on","at","to","for",
    "of","and","or","but","if","then","this","that","it","as","with",
    "by","from","up","about","into","over","after","below","above",
    "he","she","they","we","you","i","me","him","her","them","my",
    "your","his","their","our","be","been","being","have","has","had",
    "do","does","did","so","because","while","can","will","just","ur"
}

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# smart spam detection
def is_spam_rule_based(text):
    text = text.lower()

    spam_patterns = [
        r"\bdm\b",
        r"earn.*money",
        r"work.*home",
        r"free.*offer",
        r"click.*link",
        r"crypto|bitcoin",
        r"investment",
        r"telegram",
        r"verify.*account",
        r"win.*money",
        r"limited.*offer"
    ]

    for pattern in spam_patterns:
        if re.search(pattern, text):
            return True

    return False

message = st.text_area("Enter message:")

if st.button("Check Spam"):
    if message:

        if is_spam_rule_based(message):
            st.error("❌ Spam Message (rule-based detection — confidence: 0.99)")
        else:
            msg_clean = clean_text(message)
            msg_vec = vectorizer.transform([msg_clean])

            prob = model.predict_proba(msg_vec)[0][1]

            if model.predict(msg_vec)[0] == 1:
                st.error(f"❌ Spam Message (confidence: {prob:.2f})")
            else:
                st.success(f"✅ Not Spam (confidence: {1-prob:.2f})")

    else:
        st.warning("Please enter a message.")
