import pickle
import string
import re

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

# -------------------------------
# STOPWORDS
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
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------------------
# SMART SPAM RULE DETECTOR ⭐
# -------------------------------
def is_spam_rule_based(text):
    text = text.lower()

    spam_patterns = [
        r"\bdm\b",            # dm me / dm now / dm for fun
        r"earn.*money",       # earn money / earn some money
        r"work.*home",        # work from home
        r"free.*offer",       # free offer
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


print("\nType message to test spam detection\n")

while True:
    msg = input("Enter message (exit to stop): ")

    if msg.lower() == "exit":
        break

    # RULE-BASED CHECK FIRST
    if is_spam_rule_based(msg):
        print("❌ Spam (rule-based detection — confidence: 0.99)")
        continue

    # ML PREDICTION
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])

    prob = model.predict_proba(msg_vec)[0][1]

    if model.predict(msg_vec)[0] == 1:
        print("❌ Spam (confidence:", round(prob,2),")")
    else:
        print("✅ Not Spam (confidence:", round(1-prob,2),")")
