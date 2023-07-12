import string
import cv2
import numpy as np


char_to_int = {char: i for i, char in enumerate(string.ascii_uppercase + string.ascii_lowercase + \
                                                string.digits + string.punctuation + ' ')}


def get_char_to_int():
    return char_to_int


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 170, 255, cv2.THRESH_OTSU)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1).transpose((1, 0, 2))

    return image


def text_to_sequence(text, max_label_len):
    sequence = [char_to_int[char] for char in text]
    sequence += [-1] * (max_label_len - len(sequence))

    return sequence


def prepare_training_data(train_data_paths, train_data_labels):
    input_data = []
    output_data = []

    max_label_len = len(max(train_data_labels, key=len))

    for path, label in zip(train_data_paths, train_data_labels):
        image = load_image(path)
        input_data.append(image)

        sequence = text_to_sequence(label, max_label_len)
        label_len = len(label)
        input_len = image.shape[0] // 4

        sequence_arr = np.array(sequence, dtype=np.int32)
        label_len_arr = np.full((max_label_len,), label_len, dtype=np.int32)
        input_len_arr = np.full((max_label_len,), input_len, dtype=np.int32)

        packed_arr = np.stack([sequence_arr, label_len_arr, input_len_arr], axis=1)

        output_data.append(packed_arr)

    input_data = np.array(input_data)
    output_data = np.array(output_data)

    return input_data, output_data