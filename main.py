# Importing the required libraries
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow import Tokenizer
from tensorflow import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from generate_data import generate_fake_data
from train_model import train_model
from database_connection import connect_to_database

df = pd.read_csv('Coursera_data.csv')
df2 = pd.read_csv('Company_materials.csv')
data = pd.concat([df, df2], axis=0, ignore_index=True)
features = data['Text']
labels = data['Label']

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)
padded = pad_sequences(sequences, padding='post')

x_train, x_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(5000, 128, input_length=padded.shape[1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

predictions = model.predict(x_test)
predictions = [1 if p > 0.5 else 0 for p in predictions]

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

joblib.dump(model, 'your_model.joblib')

generate_fake_data()

data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2']]
y = data['target']

train_model(X, y)

connect_to_database()