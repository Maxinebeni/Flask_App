import joblib
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.cluster.util import cosine_distance
import nltk
from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

nltk.download('punkt')

app = Flask(__name__)
CORS(app)  # Allow requests from all origins

# Load your trained model
model = joblib.load('text_classification_model.pkl')

def summarize_text(text):
    # Tokenize sentences
    sentences = sent_tokenize(text)

    # Calculate sentence embeddings using TF-IDF or other methods
    # Here, we'll use a simple bag-of-words approach
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    # Convert the count matrix to TF-IDF representation
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X)
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(X_tfidf, X_tfidf)
    
    # Apply TextRank algorithm
    sentence_scores = similarity_matrix.sum(axis=1)
    sentence_scores_normalized = (sentence_scores - min(sentence_scores)) / (max(sentence_scores) - min(sentence_scores))
    sentence_scores_normalized += 0.1  # Add a small number to avoid zero-division
    max_sentence_scores = max(sentence_scores_normalized)
    sentence_scores_normalized = sentence_scores_normalized / max_sentence_scores

    scored_sentences = sorted(enumerate(sentence_scores_normalized), key=lambda x: (x[1], x[0]), reverse=True)

    summary = ''
    for i, score in scored_sentences[:3]:
        summary += ' ' + (sentences[i])

    return summary

# Function to extract text from a PDF file
def get_text_from_pdf(uploaded_file):
    try:
        pdf_reader = pdfplumber.open(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return str(e)

# Function to extract text from a URL using Beautiful Soup
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        main = soup.find('main')
        paragraphs = main.find_all('p')

        text = ' '.join([paragraph.get_text() for paragraph in paragraphs])

        if not text.strip():
            raise ValueError("Unable to determine content from the website.")

        return text
    except Exception as e:
        return str(e)

# Function to classify text using the trained model
def classify_text(text):
    # Your code for model prediction goes here
    # Assume `model.predict(text)` returns 1 for health-related and 0 for non-health-related
    prediction = model.predict([text])[0]
    return prediction, text

# Function to generate summary from text
def generate_summary(text):
    summary = summarize_text(text)
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    title = request.form['title']
    input_type = request.form['input_type']
    
    if input_type == 'url':
        url_link = request.form['url_link']
        text = get_text_from_url(url_link)
    elif input_type == 'pdf':
        pdf_file = request.files['pdf_file']
        text = get_text_from_pdf(pdf_file)

    prediction, text = classify_text(text)
    
    # Prepare the message based on the prediction
    if prediction == 1:
        message = "The article is health-related."
        summary = summarize_text(text)
    else:
        message = "The article is not health-related."
        summary = ""

    # Return the message and summary as JSON response
    return jsonify({'prediction_message': message, 'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
