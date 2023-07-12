import json
from data_preprocessing import load_image, get_char_to_int
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
from train import TrainModel
import numpy as np
from tensorflow import keras
import Levenshtein as lev


def cer(reference, hypothesis):
    reference = reference.replace(' ', '')
    hypothesis = hypothesis.replace(' ', '')
    edit_distance = lev.distance(reference, hypothesis)
    return float(edit_distance) / len(reference)


class TuneModel:
    def __init__(self, data_directory):

        with open(data_directory, 'r') as json_file:
            test_data = json.load(json_file)['train']

        self.test_imgs = [load_image(item['path']) for item in test_data]
        self.test_labels = [item['text'] for item in test_data]

        char_to_int = get_char_to_int()
        self.int_to_char = {i: char for char, i in char_to_int.items()}

    def objective(self, params):

        train_model = TrainModel()
        model = train_model.train_model(params)

        predictions = model.predict(self.test_imgs)
        input_length = np.ones(predictions.shape[0]) * predictions.shape[1]
        decoded_sequences = keras.backend.get_value(keras.backend.ctc_decode(predictions, input_length, greedy=True)[0][0])
        predicted_texts = [''.join([self.int_to_char[int(p)] for p in sequence if int(p) != -1]) for sequence in decoded_sequences]

        loss = np.mean([cer(label, predicted_text) for label, predicted_text in zip(self.test_labels, predicted_texts)])

        return {'loss': loss, 'status': STATUS_OK, 'model': model}

    def tune_model(self, param_space, max_evals=100, early_stop=15, verbose=True):

        trials = Trials()

        try:
            fmin(fn=self.objective,
                 space=param_space,
                 algo=tpe.suggest,
                 max_evals=max_evals,
                 trials=trials,
                 early_stop_fn=no_progress_loss(early_stop),
                 verbose=verbose)

        except KeyboardInterrupt:
            pass

        return trials.best_trial['result']['model']


if __name__ == '__main__':

    param_space = {
        'network_parameters': {
            'input_shape': (1024, 64, 1),
            'output_shape': len(get_char_to_int()) + 1,
            'conv_filters': hp.choice('conv_filters', [[(8, 16), (32, 64)], [(16, 32), (64, 128)], [(32, 64), (128, 256)]]),
            'lstm_units': hp.choice('lstm_units', [[i, i] for i in range(64, 128, 16)]),
            'learning_rate': hp.choice('learning_rate', [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),},
        'epochs': 10,#hp.choice('epochs', [10, 15, 20]),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        'verbose': True
    }

    tune_model = TuneModel('data_structure.json')
    model = tune_model.tune_model(param_space=param_space, max_evals=10, early_stop=3, verbose=True)