import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()


class model:
    sequantial_object: object
    tokenizer: object


class PNModel:
    ratio: object
    result: object


def train_data(df):
    rating = df['sentiment']

    le = LabelEncoder()

    rating = le.fit_transform(rating)

    rating = rating.tolist()

    comments = df['review'].values.tolist()

    maxWords = 10000

    model.tokenizer = Tokenizer(num_words=maxWords)

    model.tokenizer.fit_on_texts(comments)  # tokenleştirme işlemi

    comments = model.tokenizer.texts_to_sequences(comments)  # oluşan token değerleri karşılık gelen kelimelere atandı

    num_tokens = [len(tokens) for tokens in comments]

    mean_tokens = np.mean(np.array(num_tokens))

    std_tokens = np.std(np.array(num_tokens))

    max_tokens = int(mean_tokens + 2 * std_tokens)

    np.sum((np.array(num_tokens) < max_tokens) / len(num_tokens))

    comments = pad_sequences(comments, maxlen=max_tokens)  # hepsini ortak değerde eşitledik

    model.sequantial_object = tf.keras.Sequential()

    embedding_size = 50;  # her kelimeye karşılık gelen 50 uzunluğunda vektor

    model.sequantial_object.add(Embedding(input_dim=maxWords,
                                          output_dim=embedding_size,
                                          input_length=max_tokens,
                                          name='embedding_layer'))

    model.sequantial_object.add(
        CuDNNGRU(units=16, return_sequences=True))  # return sequence output olarak sequnce tamamı çalışıyor
    model.sequantial_object.add(CuDNNGRU(units=8, return_sequences=True))
    model.sequantial_object.add(CuDNNGRU(units=4))  # retun_sequnces=false

    model.sequantial_object.add(Dense(1, activation='sigmoid'))  # 1 ile 0 arasında değer oluşturuyor

    optimizerValue = 1e-3

    my_optimizer = tf.keras.optimizers.Adam(lr=optimizerValue)  # loss değerini belirleyecek optimizer algoritması

    # sadece pozitif veya negatif değer dönüceği için binary cross entropy kullandık
    # metrics accuracy başarı oranını gösteriyor

    model.sequantial_object.compile(loss='binary_crossentropy',
                                    optimizer=my_optimizer,
                                    metrics=(['accuracy']))

    model.sequantial_object.summary()

    tf.debugging.set_log_device_placement(True)
    with tf.device('/GPU:0'):
        model.sequantial_object.fit(comments, np.array(rating), epochs=1,
                                    batch_size=256)  # epoch => iterasyon sayısı batch_size => iterasyonda öğrenilcek eleman sayısı
    # eğitim sırasında gözüktüğü üzere loss düşerken accuracy artıyor


class ReadText(Resource):
    def post(self):
        request_data = request.form

        text = [request_data['text']]

        # parser.add_argument("text")
        # args = parser.parse_args()
        # text = [args["text"]]

        my_token = model.tokenizer.texts_to_sequences(text)
        PNModel.ratio = float(model.sequantial_object.predict(x=my_token)[0])
        if PNModel.ratio > 0.5:
            PNModel.result = "positive"
        else:
            PNModel.result = "negative"
        return jsonify({"ratio": PNModel.ratio, "result": PNModel.result})


api.add_resource(ReadText, '/text')

if __name__ == "__main__":
    dataset = pd.read_csv("IMDBDataset.csv")
    train_data(dataset)

    app.run(port=3000)
