import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

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


# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Tokenization function
def tokenize_data(data, max_length=128):
    return tokenizer(
        data['text'].tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Tokenize datasets
X_train = tokenize_data(train_data)
X_val = tokenize_data(val_data)
X_test = tokenize_data(test_data)

# Prepare TensorFlow dataset objects
train_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': X_train['input_ids'], 
        'attention_mask': X_train['attention_mask'], 
        'token_type_ids': X_train['token_type_ids']
    },
    y_train
)).batch(32).shuffle(1000)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': X_val['input_ids'], 
        'attention_mask': X_val['attention_mask'], 
        'token_type_ids': X_val['token_type_ids']
    },
    y_val
)).batch(32)

test_dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': X_test['input_ids'], 
        'attention_mask': X_test['attention_mask'], 
        'token_type_ids': X_test['token_type_ids']
    },
    y_test
)).batch(32)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy('accuracy')]
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)
