import os
from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pdfplumber
import joblib
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from flask_cors import CORS # Import CORS from flask_cors module


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes of the app


# Load your trained model
model = joblib.load('htext_classification_model.pkl')

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
        
        paragraphs = soup.find_all('p') # Adjust this based on the HTML structure
        
        text = ' '.join([paragraph.get_text() for paragraph in paragraphs])
        
        # Check if text is empty after extraction
        if not text.strip():
            raise ValueError("Unable to determine content from the website.")
        
        return text
    except Exception as e:
        return str(e)

# Function to classify text using the trained model
def classify_text(text):
    # Your code for model prediction goes here
    # Assume `model.predict(text)` returns 1 for health-related and 0 for non-health-related
    return model.predict([text])

# Function to generate summary using LexRank
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count=6)  # Adjust the number of sentences for summary
    return ' '.join([str(sentence) for sentence in summary])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.form  # Use request.form instead of request.get_json() to handle form data
    
    option = data.get('input_type')
    text = ""  # Initialize text variable with an empty string


    if option == 'url':
        url_link = data.get('url_link')
        text = get_text_from_url(url_link)

        if not text or isinstance(text, str) and text.startswith("<!doctype"):
            return jsonify({
                'error': 'Unable to extract text from the provided URL. Please ensure the URL is valid and contains readable content.'
            }), 500
            
    elif option == 'pdf':
        pdf_file = request.files['pdf_file']
        text = get_text_from_pdf(pdf_file)

        if not text:
            return jsonify({
                'error': 'Failed to process the PDF file. Please make sure it is a valid PDF.'
            }), 500

    prediction = classify_text(text)

    if prediction == 1:
        message = "The article is health-related."
        summary = generate_summary(text)
    else:
        message = "The article is not health-related."
        summary = None

    return jsonify({
        'prediction_message': message,
        'summary': summary
    }), 200  # Change the status code to 200

if __name__ == '__main__':
    app.run(debug=True)
