import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

    # remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])

    # remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# -------------------------------
# LOAD DATASET
# -------------------------------
print("Loading dataset...")

data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# convert labels
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# clean text
data["message"] = data["message"].apply(clean_text)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# -------------------------------
# IMPROVED TF-IDF (BIGRAM + TRIGRAM)
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),   # unigram + bigram + trigram ⭐
    max_features=5000,   # keep important words
    min_df=2             # ignore rare words
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -------------------------------
# CHECK ACCURACY
# -------------------------------
pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, pred))

# -------------------------------
# TEST CUSTOM MESSAGE
# -------------------------------
print("\nType a message to check if it's spam.")

while True:
    msg = input("\nEnter message (or type 'exit'): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])

    prob = model.predict_proba(msg_vec)[0][1]  # spam probability

    if model.predict(msg_vec)[0] == 1:
        print(f"❌ Spam Message (confidence: {prob:.2f})")
    else:
        print(f"✅ Not Spam (confidence: {1-prob:.2f})")
