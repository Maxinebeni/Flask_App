import joblib
import io
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.cluster.util import cosine_distance
import nltk
import requests
from bs4 import BeautifulSoup
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

nltk.download('punkt')

app = Flask(__name__)
CORS(app)

model = joblib.load('text_classification_model.pkl')

def summarize_text(text):
    sentences = sent_tokenize(text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X)

    similarity_matrix = cosine_similarity(X_tfidf, X_tfidf)

    sentence_scores = similarity_matrix.sum(axis=1)
    sentence_scores_normalized = (sentence_scores - min(sentence_scores)) / (max(sentence_scores) - min(sentence_scores))
    sentence_scores_normalized += 0.1
    max_sentence_scores = max(sentence_scores_normalized)
    sentence_scores_normalized = sentence_scores_normalized / max_sentence_scores

    scored_sentences = sorted(enumerate(sentence_scores_normalized), key=lambda x: (x[1], x[0]), reverse=True)

    summary = ""
    for i, score in scored_sentences[:3]:
        summary += ' ' + sentences[i]

    return summary

def get_text_from_pdf(uploaded_file):
    if uploaded_file.content_type not in ['application/pdf']:
        return "Invalid file type. Please upload a PDF file."

    try:
        pdf_reader = pdfplumber.open(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return str(e)

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

def classify_text(text):
    prediction = model.predict([text])[0]
    return prediction, text

def extract_pdf_text(pdf_file):
    try:
        pdf_reader = pdfplumber.open(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception:
        return None

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

        if not text or isinstance(text, str) and text.startswith("<!doctype"):
            return jsonify({
                'error': 'Unable to extract text from the provided URL. Please ensure the URL is valid and contains readable content.'
            }), 500
    elif input_type == 'pdf':
        pdf_file = request.files['pdf_file']
        text = extract_pdf_text(pdf_file)

        if text is None:
            return jsonify({
                'error': 'Failed to process the PDF file. Please make sure it is a valid PDF.'
            }), 500

    prediction, text = classify_text(text)

    if prediction == 1:
        message = "The article is health-related."
        summary = summarize_text(text)
    else:
        message = "The article is not health-related."
        summary = ""

    return jsonify({
        'prediction_message': message,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(debug=True)
