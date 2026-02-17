import pandas as pd
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Training model...")

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
    text = str(text).lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1","v2"]]
data.columns = ["label","message"]

data["label"] = data["label"].map({"ham":0,"spam":1})
data["message"] = data["message"].apply(clean_text)

# -------------------------------
# SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label"], test_size=0.2, random_state=42
)

# -------------------------------
# TFIDF FEATURES
# -------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    max_features=8000,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# MODEL (LOGISTIC REGRESSION)
# -------------------------------
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000
)

model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
print("Model Accuracy:", accuracy_score(y_test, pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("✅ Model saved successfully!")
