import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_data = pd.read_csv('./training.csv')
val_data = pd.read_csv('./validation.csv')
test_data = pd.read_csv('./test.csv')

# Basic preprocessing
def preprocess_text(df):
    df['cleaned_text'] = df['text'].str.lower().str.replace(r'[^\w\s]', '')
    return df

train_data = preprocess_text(train_data)
val_data = preprocess_text(val_data)
test_data = preprocess_text(test_data)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['label'])
y_val = label_encoder.transform(val_data['label'])
y_test = label_encoder.transform(test_data['label'])

X_train = train_data['cleaned_text']
X_val = val_data['cleaned_text']
X_test = test_data['cleaned_text']