from network import Network
from data_preprocessing import prepare_training_data, get_char_to_int, load_image
import json


class TrainModel:

    def __init__(self, data_directory='data_structure.json'):

        with os.open(data_directory, 'r') as json_file:
            train_data = json.load(json_file)['train']

        train_data_paths = [item['path'] for item in train_data]
        train_data_labels = [item['text'] for item in train_data]

        self.input_data, self.output_data = prepare_training_data(train_data_paths, train_data_labels)

    def train_model(self, network_parameters, epochs=15, batch_size=32, verbose=1):
        network = Network(**network_parameters)
        model = network.model()

        model.fit(self.input_data, self.output_data, epochs=epochs, batch_size=batch_size, verbose=verbose)

        return model


if __name__ == '__main__':

    model_parameters = {
        'input_shape': (1024, 64, 1),
        'output_shape': len(get_char_to_int()) + 1,
        'conv_filters': [(8, 16), (32, 64)],
        'lstm_units': [64, 64],
        'learning_rate': 0.001,
    }

    train_model = TrainModel()
    model = train_model.train_model(model_parameters, epochs=100, batch_size=32)

    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/model.keras')

