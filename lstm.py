import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize

# Corrected modern TensorFlow/Keras imports
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input

# Import the text data from your doc.py file
from doc import training_data

# 1. Clean the text from doc.py
cleaned = re.sub(r'\W+', ' ', training_data).lower()
tokens = word_tokenize(cleaned)

# 2. Slice text into windows of 4 words
train_len = 3 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

# 3. Vectorize text into unique structural integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 

vocabulary_size = len(tokenizer.word_counts) + 1
n_sequences = np.empty([len(sequences), train_len], dtype='int32')

for i in range(len(sequences)):
    n_sequences[i] = sequences[i]

# Split data into X (inputs) and y (targets)
train_inputs = n_sequences[:, :-1]
train_targets = n_sequences[:, -1]
seq_len = train_inputs.shape[1]

# 4. Build the Deep Learning Neural Network Architecture
model = Sequential() # creates nn layer by layer 
model.add(Input(shape=(seq_len,))) 
model.add(Embedding(vocabulary_size, seq_len))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(vocabulary_size, activation='softmax'))

print(model.summary())

# 5. Compile and run training loop
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Reduced epochs to 50 for quick testing; change back to 500 for maximum accuracy
model.fit(train_inputs, train_targets, epochs=50, batch_size=64, verbose=1)
model.save("mymodel.h5")

# 6. Interactive User Next-Word Prediction Loop
print("\n--- Model training complete! ---")
input_text = input("Enter a phrase (around 3 words): ").strip().lower()
encoded_text = tokenizer.texts_to_sequences([input_text])

if not encoded_text[0]:
    print("Words not found in the vocabulary!")
else:
    pad_encoded = pad_sequences(encoded_text, maxlen=seq_len, truncating='pre')
    predictions = model.predict(pad_encoded, verbose=0)
    for i in predictions[0].argsort()[-3:][::-1]:
        pred_word = tokenizer.index_word[i]
        print("The next word can be:", pred_word)
