# resume_screener_model.py

import pandas as pd
import re
import os
import pickle
import torch
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/UpdatedResumeDataSet.csv")  # Columns: resume_text, category

# Initialize components
lemmatizer = WordNetLemmatizer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

df['cleaned'] = df['Resume'].apply(clean_text)

# Get BERT Embedding
def get_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

df['embedding'] = df['cleaned'].apply(get_bert_embedding)
X = list(df['embedding'])
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Save model and tokenizer
os.makedirs('model', exist_ok=True)
with open("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/model/bert_classifier.pkl", 'wb') as f:
    pickle.dump(clf, f)
with open("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/model/bert_tokenizer.pkl", 'wb') as f:
    pickle.dump(tokenizer, f)
with open("C:/Users/Zaid Chikte/Desktop/AI-Powered Resume Screener (NLP + ML)/model/bert_model.pkl", 'wb') as f:
    pickle.dump(bert_model, f)

print("✅ Model and BERT components saved successfully.")