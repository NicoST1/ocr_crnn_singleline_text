# Single-Line Text OCR

Optical Character Recognition (OCR) tool, designed specifically for single-line text images.

## Core Components

- **CNN (Convolutional Neural Network)**: Used for feature extraction from images.
- **Time-distributed Layer**: Applies the same operation to each timestep in a sequence.
- **LSTM (Long Short-Term Memory)**: Deals with sequence data and preserving context.
- **Dense Layer**: Handles complex pattern recognition.
- **CTC (Connectionist Temporal Classification)**: Manages the alignment between input images and their corresponding textual outputs.

## Usage

To run the model, execute the `evaluate.py` file. This will produce a window displaying the original image and the predicted text. You can navigate through the images and close the window at your convenience. A quantitative summary of results will be provided in the terminal.

If you're interested in training your own model, you'll first need to generate your own dataset by running the `generate_data.py` file. Follow this up by executing the `train.py` script. If you wish to tune hyperparameters, use `tune_model.py`. Be aware, this process can be time-consuming!
