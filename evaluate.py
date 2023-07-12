from tensorflow import keras
from data_preprocessing import load_image, get_char_to_int
import json
import numpy as np
from tune_model import cer
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Button
import cv2


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind >= len(validate_imgs):
            self.ind = 0
        ax.clear()
        ax.imshow(validate_imgs[self.ind], cmap='gray')
        ax.set_title(predicted_texts[self.ind])
        ax.axis('off')
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        if self.ind < 0:
            self.ind = len(validate_imgs) - 1
        ax.clear()
        ax.imshow(validate_imgs[self.ind], cmap='gray')
        ax.set_title(predicted_texts[self.ind])
        ax.axis('off')
        plt.draw()



if __name__ == '__main__':

    model = keras.models.load_model('models/model.keras', compile=False)

    with open('validation_structure.json', 'r') as json_file:
        validate_data = json.load(json_file)

    validate_imgs = np.array([load_image(item['path']) for item in validate_data])
    validate_labels = [item['text'] for item in validate_data]

    int_to_char = {i: char for char, i in get_char_to_int().items()}

    predictions = model.predict(validate_imgs)
    input_length = np.ones(predictions.shape[0]) * predictions.shape[1]
    decoded_sequences = keras.backend.get_value(keras.backend.ctc_decode(predictions, input_length, greedy=True)[0][0])
    predicted_texts = [''.join([int_to_char[int(p)] for p in sequence if int(p) != -1]) for sequence in decoded_sequences]

    loss = np.mean([cer(label, predicted_text) for label, predicted_text in zip(validate_labels, predicted_texts)])
    accuracy = np.mean([1 if label == predicted_text else 0 for label, predicted_text in zip(validate_labels, predicted_texts)])
    validate_imgs = validate_imgs.astype(np.float32) / 255.0
    validate_imgs = [cv2.resize(img.transpose((1, 0, 2)), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for img in validate_imgs]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    ax.imshow(validate_imgs[0], cmap='gray')
    ax.set_title(predicted_texts[0])
    ax.axis('off')
    plt.show()

    plt.close()

    print(f"Avg. Character Error Rate: {loss}, Accuracy: {accuracy}")


