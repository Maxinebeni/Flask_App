from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pdfplumber
import validators
import joblib
import re

app = Flask(__name__)

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

        if "example1.com" in url:
            paragraphs = soup.find_all('p')
        elif "example2.com" in url:
            paragraphs = soup.find_all('div')
        else:
            paragraphs = soup.find_all('p')
        
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
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    # Take the first few sentences as the summary
    summary = ' '.join(sentences[:3])  # Change 3 to the desired number of sentences
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
        summary = generate_summary(text)
    else:
        message = "The article is not health-related."
        summary = ""

    # Return the message and summary as JSON response
    return jsonify({'prediction_message': message, 'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
