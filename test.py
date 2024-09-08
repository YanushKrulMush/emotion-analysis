import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load data
train_data = pd.read_csv('./goemotions/train.tsv', sep='\t', header=None)
val_data = pd.read_csv('./goemotions/dev.tsv', sep='\t', header=None)
test_data = pd.read_csv('./goemotions/test.tsv', sep='\t', header=None)

# Columns: 0 is text, 1 is label(s)
train_texts = train_data[0]
train_labels = train_data[1]
val_texts = val_data[0]
val_labels = val_data[1]
test_texts = test_data[0]
test_labels = test_data[1]

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply preprocessing to train, validation, and test sets
train_texts = train_texts.apply(preprocess_text)
val_texts = val_texts.apply(preprocess_text)
test_texts = test_texts.apply(preprocess_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)
X_test = vectorizer.transform(test_texts)

# Initialize and train the Logistic Regression model using One-vs-Rest strategy
clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, train_labels)

# Predict on validation set
val_preds = clf.predict(X_val)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(val_labels, val_preds)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='macro')

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1-Score: {val_f1:.4f}")

# Predict on test set
test_preds = clf.predict(X_test)

# Evaluate the model on the test set
test_accuracy = accuracy_score(test_labels, test_preds)
test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='macro')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
