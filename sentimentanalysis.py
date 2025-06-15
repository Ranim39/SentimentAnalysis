import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

word_index =imdb.get_word_index()

x_train.shape

x_test.shape

y_train[0]

word_index = imdb.get_word_index()
reverse_word_index = {value+3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in x_train[0]])
print(decoded_review)

x_train = pad_sequences(x_train, maxlen=256)
x_test = pad_sequences(x_test, maxlen=256)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=256),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  
])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=18, batch_size=512, validation_split=0.2)

model.evaluate(x_test, y_test)

y_predicted=model.predict(x_test)

x_test[0]

y_predicted[0]

y_test[0]

y_predicted[9]

y_test[9]

def encode_review(text):
    words = text.lower().split()  
    encoded = [word_index.get(word, 2) for word in words]  
    return encoded  

from tensorflow.keras.preprocessing.sequence import pad_sequences

sample_text = "I really loved this movie, it was amazing"

encoded = encode_review(sample_text)

padded = pad_sequences([encoded], maxlen=200)

prediction = model.predict(padded)

print("Probabilité que la critique soit positive :", prediction[0][0])

if prediction[0][0] > 0.5:
    print("➡️ Critique positive")
else:
    print("⬅️ Critique négative")