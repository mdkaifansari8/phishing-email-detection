import imaplib
import email
from email.header import decode_header
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
from flask import Flask, request, jsonify

# Download NLTK resources (only need to run once)
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Email credentials and IMAP server configuration
username = "mak0786a@gmail.com"         
password = "hzlwqpbysapyzxzp"              
imap_server = "imap.gmail.com"              
# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset
df = pd.read_csv('Phishing_Email.csv')
df['Email Text'] = df['Email Text'].fillna('')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess function
def preprocess_email(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the dataset
df['processed_text'] = df['Email Text'].apply(preprocess_email)
df['Email Type'] = df['Email Type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)

# TF-IDF Vectorization and Model Training
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['Email Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear'], 'max_iter': [500]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from GridSearch
best_model = grid_search.best_estimator_

# Function to predict phishing emails
def predict_new_email(emails):
    processed_emails = [preprocess_email(email) for email in emails]
    features = vectorizer.transform(processed_emails).toarray()
    predictions = best_model.predict(features)
    return ["Phishing" if pred == 1 else "Safe" for pred in predictions]

# Function to fetch emails from the inbox
def fetch_emails(folder="inbox", num_emails=5):
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(username, password)
    mail.select(folder)

    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()
    emails = []

    for i in email_ids[-num_emails:]:
        status, msg_data = mail.fetch(i, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")

                # Check if the email is multipart
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))

                        # Check if payload is not None before decoding
                        payload = part.get_payload(decode=True)
                        if payload and content_type == "text/plain" and "attachment" not in content_disposition:
                            try:
                                body = payload.decode()  # Decode only if payload is valid
                                emails.append(body)
                            except Exception as e:
                                print(f"Could not decode email body: {e}")
                else:
                    # Handle non-multipart emails
                    payload = msg.get_payload(decode=True)
                    if payload:  # Check if payload is not None before decoding
                        try:
                            body = payload.decode()
                            emails.append(body)
                        except Exception as e:
                            print(f"Could not decode email body: {e}")

    mail.close()
    mail.logout()

    return emails


# Flask route to fetch and predict phishing emails
@app.route('/fetch_and_predict', methods=['GET'])
def fetch_and_predict():
    try:
        emails = fetch_emails(num_emails=10)
        predictions = predict_new_email(emails)
        results = [{"email": email, "prediction": prediction} for email, prediction in zip(emails, predictions)]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app locally
if __name__ == '__main__':
    app.run(port=5000, debug=True)

# setup
# restart your Flask server:  python app.py

# paste the link on the below browser
# http://127.0.0.1:5000/fetch_and_predict

