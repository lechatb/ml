from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# массив слов
words = ["cat", "dog", "bird", "fish", "horse", "chicken"]

# преобразование слов в векторы
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(words)
X = (X.T / np.amax(X.T, axis=0)).T # нормализация

# размер кодированного представления
encoding_dim = 2

# входной плейсхолдер
input_word = Input(shape=(len(X[0]),))
# кодированное представление слова
encoded = Dense(encoding_dim, activation='relu')(input_word)
# декодированное представление слова
decoded = Dense(len(X[0]), activation='sigmoid')(encoded)

# автоэнкодер
autoencoder = Model(input_word, decoded)

# энкодер
encoder = Model(input_word, encoded)

# компиляция модели
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# обучение автоэнкодера
autoencoder.fit(X, X, epochs=50, batch_size=2, shuffle=True)

# получаем закодированные представления слов
encoded_words = encoder.predict(X)

# выводим закодированные и декодированные представления слов
for i, word in enumerate(words):
    print("Word:", word)
    print("Encoded representation:", encoded_words[i])
    print("Decoded representation:", autoencoder.predict(X)[i])
```

В данном примере используется библиотека Keras и простой автоэнкодер с одним скрытым слоем. Функция принимает массив слов, преобразует его в