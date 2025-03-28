import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob

# 🔹 Step 1: Load and Expand Dataset
data = [
    {"news": "NASA successfully lands a new rover on Mars to search for life.", "source": "cnn.com", "label": 1},
    {"news": "Aliens secretly control the White House", "source": "fake-news.com", "label": 0},
    {"news": "The UN launches a global climate initiative", "source": "bbc.com", "label": 1},
    {"news": "Government to replace officials with AI robots", "source": "unknown-source.com", "label": 0},
    {"news": "New study finds coffee increases lifespan", "source": "nytimes.com", "label": 1},
    {"news": "Moon is actually a hologram created by aliens", "source": "fake-news.com", "label": 0},
    {"news": "COVID-19 vaccines are effective and reduce transmission", "source": "who.int", "label": 1},
    {"news": "5G towers cause coronavirus", "source": "fake-news.net", "label": 0},
    {"news": "Stock markets hit record high after economic growth", "source": "reuters.com", "label": 1},
    {"news": "The earth is flat and NASA is hiding the truth", "source": "conspiracy.com", "label": 0},
]

df = pd.DataFrame(data)

# 🔹 Step 2: Assign Source Credibility Scores
credibility_scores = {
    "cnn.com": 0.9, 
    "bbc.com": 0.9, 
    "reuters.com": 0.9,
    "nytimes.com": 0.85, 
    "guardian.com": 0.8, 
    "foxnews.com": 0.7,
    "who.int": 0.95, 
    "unknown-source.com": 0.2, 
    "fake-news.com": 0.05,
    "fake-news.net": 0.05, 
    "conspiracy.com": 0.1,
    # Indian News Sources
    "hindustantimes.com": 0.85,  # Established credible news source
    "timesofindia.indiatimes.com": 0.85,  # Major Indian news site
    "ndtv.com": 0.85,  # Reliable Indian news outlet
    "thehindu.com": 0.9,  # Well-respected Indian newspaper
    "indianexpress.com": 0.85,  # Credible Indian news source
    "tribuneindia.com": 0.8,  # Established Indian news source
    "moneycontrol.com": 0.8,  # Business-focused but reliable
    "livemint.com": 0.8,  # Economic and business news outlet
    "dnaindia.com": 0.75,  # Known Indian news outlet
    "firstpost.com": 0.75,  # Major news site
    "news18.com": 0.8,
}

df["credibility"] = df["source"].apply(lambda x: credibility_scores.get(x, 0.5))

# 🔹 Step 3: Clean and Process Text
nltk.download("punkt")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df["clean_news"] = df["news"].apply(clean_text)

# 🔹 Step 4: Compute Sentiment Scores
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Positive/Negative emotion score

df["sentiment"] = df["clean_news"].apply(get_sentiment)

# 🔹 Step 5: Extract Features (Increased max_features for better learning)
vectorizer = TfidfVectorizer(max_features=200)
X_tfidf = vectorizer.fit_transform(df["clean_news"]).toarray()
X = np.hstack((X_tfidf, df[["credibility", "sentiment"]].values))
y = df["label"]

# 🔹 Step 6: Train the Model with Optimized Parameters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 🔹 Evaluate Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 🔹 Save Model & Vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("💾 Model and vectorizer saved!")

# 🔹 User Input Function
def predict_news():
    news_text = input("📰 Enter news text: ")
    source = input("🔍 Enter news source (e.g., cnn.com): ")
    credibility = credibility_scores.get(source, 0.5)
    sentiment = get_sentiment(news_text)
    news_tfidf = vectorizer.transform([clean_text(news_text)]).toarray()
    features = np.hstack((news_tfidf, [[credibility, sentiment]]))
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100
    result = "Real News" if prediction == 1 else "Fake News"
    print(f"\n📰 News: {news_text}\n🔍 Source: {source}\n🧐 Confidence: {confidence:.2f}%\n✅ Prediction: {result}\n")

# 🔹 Run the User Input Function
predict_news()
