import pandas as pd
import numpy as np
import re
import nltk
from textblob import TextBlob
from newspaper import Article
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure NLTK resources are downloaded safely
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK resources: {e}")

# Load sentence transformer model with error handling
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

# Updated credibility scores
credibility_scores = {
    "cnn.com": 0.9, 
    "bbc.com": 0.9, 
    "reuters.com": 0.9,
    "nytimes.com": 0.85, 
    "hindustantimes.com": 0.85,
    "timesofindia.indiatimes.com": 0.85,  
    "ndtv.com": 0.85,
    "indianexpress.com": 0.85,
    "news18.com": 0.8,
    "default": 0.1
}

# Text cleaning function
def clean_text(text):
    if not text:
        return ""
    try:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

# Sentiment analysis
def get_sentiment(text):
    if not text or text.strip() == "":
        return 0
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0

# Find relevant articles on the homepage
def find_relevant_articles(url, news_text):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Error accessing {url}: Status code {response.status_code}")
            return []
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
        # Extract news keywords (first 5 words)
        keywords = clean_text(news_text).split()[:5]
        # Find all links that might be news articles
        article_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            # Make absolute URL if relative
            if href and not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            # Check if link text contains any of our keywords
            link_text = link.get_text().lower()
            if any(keyword in link_text for keyword in keywords if keyword):
                if href and '/news/' in href or '/article/' in href or '/india/' in href or '/world/' in href:
                    article_links.append(href)
        return article_links[:5]  # Return top 5 matching links
    except Exception as e:
        print(f"Error finding articles: {e}")
        return []

# Improved Similarity Check with better article finding
def check_similarity(news_text, url, source):
    url = str(url)
    if not url:
        return 0.0
    try:
        # Clean the input news text for comparison
        clean_news_text = clean_text(news_text)
        keywords = clean_news_text.split()[:5]
        # If URL is just a homepage, find relevant articles
        if url.endswith('.com/') or url.endswith('.in/') or url.endswith('.org/') or '/news/' not in url:
            print("Searching homepage for relevant articles...")
            article_urls = find_relevant_articles(url, news_text)     
            # If we found articles, check them all
            if article_urls:
                best_similarity = 0.0
                for article_url in article_urls:
                    print(f"Checking article: {article_url}")
                    similarity = extract_and_compare(article_url, clean_news_text)
                    if similarity > best_similarity:
                        best_similarity = similarity
                return best_similarity
        # Direct URL comparison
        return extract_and_compare(url, clean_news_text)
    except RequestException as e:
        print(f"Error fetching URL content: {e}")
        return 0.1
    except Exception as e:
        print(f"Unexpected error in similarity check: {e}")
        return 0.1

# Helper function to extract text and compare
def extract_and_compare(url, clean_news_text):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Error: Unable to access URL {url} (Status Code: {response.status_code})")
            return 0.1
        # Try newspaper3k extraction
        try:
            article = Article(url)
            article.download()
            article.parse()
            extracted_text = article.text
        except:
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            # Try to find main content
            content = soup.find(['article', 'main', 'div', 'section'])
            extracted_text = content.get_text() if content else soup.get_text()
        # Clean the extracted text
        extracted_text = clean_text(extracted_text)
        if not extracted_text:
            return 0.2
        # Use sentence embedding for comparison
        input_embedding = embedding_model.encode(clean_news_text, convert_to_tensor=True)
        extracted_embedding = embedding_model.encode(extracted_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(input_embedding, extracted_embedding).item()
        # Also check for exact phrase matches
        words = clean_news_text.split()
        exact_match_score = 0.0
        if len(words) > 5:  # Only check substantial phrases
            for i in range(len(words) - 4):
                phrase = ' '.join(words[i:i+5])
                if phrase in extracted_text:
                    exact_match_score += 0.2  # Boost score for each matching phrase
        # Combine scores
        final_score = max(similarity_score, exact_match_score)
        return max(0.0, min(final_score, 1.0))
    except Exception as e:
        print(f"Error in extract_and_compare: {e}")
        return 0.1

# Function to determine if URL belongs to claimed source
def validate_source_url(url, claimed_source):
    if not url or not claimed_source:
        return False
    # Extract domain from URL
    try:
        parsed_url = urlparse(url)
        url_domain = parsed_url.netloc.lower()
        # Remove www. if present
        if url_domain.startswith('www.'):
            url_domain = url_domain[4:] 
        # Check if claimed source appears in the URL domain
        claimed_source = claimed_source.lower()
        if claimed_source in url_domain:
            return True
        return False
    except Exception as e:
        print(f"Error validating source URL: {e}")
        return False

# Function to predict news authenticity
def predict_news():
    try:
        # Load trained model and vectorizer
        model = joblib.load("improved_fake_news_model.pkl")
        vectorizer = joblib.load("improved_tfidf_vectorizer.pkl")
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        print("Using simplified verification without ML model.")
        model = None
        vectorizer = None
    news_text = input("Enter the news content: ")
    source = input("Enter the news source (e.g., bbc.com): ")
    url = input("Enter the URL (if available, else press Enter): ")
    # Make URL mandatory for high-credibility sources
    source_credibility = credibility_scores.get(source.lower(), credibility_scores["default"])
    if source_credibility > 0.7 and not url:
        print("\nURL verification is required for content from highly trusted sources.")
        url = input("Please provide URL for verification: ")
        if not url:
            print("Cannot verify authenticity without URL. Treating with reduced credibility.")
            source_credibility = 0.5
    # Clean and preprocess the input
    clean_news_text = clean_text(news_text)
    sentiment = get_sentiment(clean_news_text)
    word_count = len(clean_news_text.split())
    # Check if URL matches claimed source
    source_url_match = validate_source_url(url, source)
    if not source_url_match and url:
        print(f"\nWarning: The URL provided does not appear to be from {source}")
        source_credibility *= 0.5
    # Get content similarity with improved algorithm
    similarity_score = check_similarity(clean_news_text, url, source) if url else 0.0
    # IMPORTANT: Lower the threshold for verification!
    # Headlines on homepages are often summaries, not exact matches
    similarity_threshold = 0.3  # Reduced from higher values
    print(f"\nContent similarity score: {similarity_score:.2f}")
    # If model is available, use it for prediction
    if model and vectorizer:
        # Vectorize the news text
        try:
            news_tfidf = vectorizer.transform([clean_news_text]).toarray()
            # Combine features for prediction with higher emphasis on similarity
            adjusted_credibility = source_credibility * 10
            adjusted_similarity = similarity_score * 30
            features = np.hstack((news_tfidf, [[adjusted_credibility, sentiment * 10, word_count / 50, adjusted_similarity]]))
            # Predict using the model
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features)[0][prediction] * 100
        except Exception as e:
            print(f"Error during model prediction: {e}")
            # Fallback to rule-based approach
            prediction = 1 if (source_credibility > 0.7 and similarity_score > similarity_threshold) else 0
            confidence = 70.0
    else:
        # No model available - use rule-based approach
        if source_credibility > 0.7 and similarity_score > similarity_threshold:
            prediction = 1  # Real news
            confidence = 70.0 + (similarity_score * 30)
        else:
            prediction = 0  # Fake news
            confidence = 80.0 - (similarity_score * 30)
    # Special handling for trusted sources with sufficient verification
    if source_credibility > 0.7 and similarity_score > similarity_threshold:
        prediction = 1  # Override to real news
        confidence = max(confidence, 75.0)
        override_reason = "Content verified from trusted source"
    elif source_credibility > 0.7 and similarity_score < similarity_threshold:
        if similarity_score > 0.15:  # Some similarity found but not enough
            prediction = 0  # Treat as suspicious
            confidence = 65.0
            override_reason = "Content partially verified but needs further checking"
        else:
            prediction = 0  # Treat as fake
            confidence = 80.0
            override_reason = "Content not found on claimed trusted source"
    else:
        override_reason = None
    # Output the result
    result = "Real News" if prediction == 1 else "Fake News"
    print("\n==============================")
    print(f"📰 **News:** {news_text}")
    print(f"🌐 **Source:** {source}")
    if url:
        print(f"🔗 **Checked against:** {url}")
        print(f"📏 **Content similarity:** {similarity_score:.2f}")
    if prediction == 1:
        print(f"📊 **Prediction:** ✅ {result} (Confidence: {confidence:.2f}%)")
    else:
        print(f"📊 **Prediction:** ❌ {result} (Confidence: {confidence:.2f}%)")
    if override_reason:
        print(f"ℹ️ **Note:** {override_reason}")
    print("==============================")

# Execute prediction
if __name__ == "__main__":
    predict_news()
