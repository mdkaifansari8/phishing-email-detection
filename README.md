Email Phishing Detection Application

Overview
This application is a Flask-based web service designed to detect phishing emails. It uses Natural Language Processing (NLP) techniques for email text preprocessing and machine learning to classify emails as either "Safe" or "Phishing." The app fetches emails from an inbox using IMAP and processes them to predict their type.

Features
Email Fetching: Connects to an email inbox via IMAP to retrieve recent emails.
Preprocessing: Cleans and processes email content using NLP methods.
Phishing Detection: Classifies emails using a Logistic Regression model trained on a phishing email dataset.
Web Interface: Exposes a Flask-based API endpoint for fetching and predicting emails.
Requirements
Libraries
Ensure the following Python libraries are installed:

IMAP & Email Processing: imaplib, email, bs4
Text Processing: nltk, re, pandas
Machine Learning: sklearn
Web Framework: flask
Environment Setup
Python 3.7 or later.
A valid email account (e.g., Gmail) with an app-specific password for IMAP access.
A phishing email dataset in CSV format (e.g., Phishing_Email.csv).
Setup Instructions
1. Install Dependencies
bash
Copy code
pip install -r requirements.txt
2. Configure Email Credentials
Update the following variables in the script:

username: Your email address.
password: Your app-specific password.
imap_server: IMAP server address (e.g., imap.gmail.com).
3. Preprocess Dataset
Place your phishing email dataset (Phishing_Email.csv) in the working directory.
Ensure the dataset has at least two columns:
Email Text: The email content.
Email Type: Labels (Phishing Email or Safe Email).
4. Train the Model
The script automatically preprocesses the dataset, vectorizes it using TF-IDF, and trains a Logistic Regression model.

Running the Application
1. Start the Flask Server
Run the script:

bash
Copy code
python app.py
2. Access the API
Open a browser or use a tool like Postman to access:

text
Copy code
http://127.0.0.1:5000/fetch_and_predict
This endpoint retrieves recent emails, processes them, and returns predictions.

API Example
Request
Method: GET
Endpoint: /fetch_and_predict
Response
json
Copy code
[
    {
        "email": "This is the body of the email.",
        "prediction": "Phishing"
    },
    {
        "email": "This email is safe and legitimate.",
        "prediction": "Safe"
    }
]
Key Functions
Email Fetching:
Connects to the IMAP server and retrieves recent emails.
Email Preprocessing:
Cleans HTML, removes links and special characters, and lemmatizes words.
Model Training:
Uses TF-IDF for feature extraction and Logistic Regression for classification.
Prediction:
Predicts whether emails are "Safe" or "Phishing."
Notes
Use app-specific passwords for email accounts with 2FA enabled.
Ensure your email provider allows IMAP access.
Update the Phishing_Email.csv file with more data to improve model accuracy.
Troubleshooting
If emails are not fetched, verify IMAP settings and credentials.
For preprocessing issues, ensure nltk resources are downloaded:
python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Authors
Author: Md Kaif Ansari[LIT2021022], Shrish Kumar[LIT2021027]
