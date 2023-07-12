from tensorflow import keras
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
def ctc_loss():
    def loss(y_true, y_pred):
        batch_labels = y_true[:, :, 0]
        label_length = y_true[:, 0, 1]
        input_length = y_true[:, 0, 2]

        label_length = tf.expand_dims(label_length, -1)
        input_length = tf.expand_dims(input_length, -1)

        return keras.backend.ctc_batch_cost(batch_labels, y_pred, input_length, label_length)
    return loss


class Network:
    def __init__(self, input_shape, output_shape, conv_filters, lstm_units, learning_rate):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate

    def convolutional(self, input_img):
        x = input_img
        for f1, f2 in self.conv_filters:
            x = keras.layers.Conv2D(f1, 3, activation="relu", padding="same")(x)
            x = keras.layers.Conv2D(f2, 3, activation="relu", padding="same")(x)
            x = keras.layers.MaxPooling2D()(x)
        return x

    def recurrent(self, tdist):
        x = tdist
        for units in self.lstm_units:
            x = keras.layers.Bidirectional(keras.layers.LSTM(units, return_sequences=True))(x)
        return x

    def model(self):
        input_img = keras.Input(shape=self.input_shape, name="input_img")
        convolution = self.convolutional(input_img)
        tdist = keras.layers.TimeDistributed(keras.layers.Flatten(), name='timedistrib')(convolution)
        recurrent = self.recurrent(tdist)
        y_pred = keras.layers.Dense(self.output_shape, name="predictions", activation='softmax')(recurrent)

        model = keras.models.Model(inputs=input_img, outputs=y_pred)
        model.compile(loss=ctc_loss(), optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))

        return model

