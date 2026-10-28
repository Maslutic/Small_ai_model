# Small_ai_model

A small, from-scratch neural network for recognising handwritten digits. The project trains a simple fully connected model on the MNIST training set and then uses it to classify image files placed in the `images/` directory.

It is deliberately lightweight: there is no machine-learning framework, no command-line abstraction layer, and no hidden training pipeline. The goal is to make the core ideas—forward propagation, loss calculation, and backpropagation—easy to read in a single file.

## What it does

- Loads the MNIST training data from the included `mnist.npz` file.
- Trains a `784 → 20 → 10` neural network for three epochs.
- Uses sigmoid activations and a mean-squared-error loss.
- Finds supported image files in `images/`, normalises them to the MNIST-style 28×28 format, and prints a predicted digit with its confidence.
- Handles both light-on-dark and dark-on-light digit images automatically.

## Requirements

- Python 3.9 or newer
- NumPy
- Pillow

Install the two dependencies with:

```bash
pip install numpy pillow
```

## Quick start

Clone or download the repository, then run the project from its root directory:

```bash
python main.py
```

The script first trains the network using `mnist.npz`. After training, it scans `images/` and prints results similar to:

```text
image.jpg: digit 7, confidence 91.4%
```

`mnist.npz` is loaded using a relative path, so make sure you run the command from the project root.

## Recognising your own images

Put one or more digit images in the `images/` folder and run `python main.py` again. The following formats are supported:

- PNG
- JPG / JPEG
- BMP
- GIF
- WebP

For the most reliable result, use an image containing one clear digit with some empty space around it. The preprocessing step converts the image to grayscale, inverts it when necessary, crops the digit, resizes it to fit within 20×20 pixels, and centres it on a 28×28 canvas.

The default `images/image.jpg` is an example input. If the directory is empty, the program simply reports that no images were found.

## Project structure

```text
.
├── main.py       # model definition, training loop, and predictions
├── utils.py      # MNIST loading and image preprocessing helpers
├── mnist.npz     # MNIST training data used by the script
└── images/       # images to classify
```

## How the model works

Each 28×28 image is flattened into 784 pixel values. Those values pass through a hidden layer of 20 neurons and an output layer of 10 neurons—one for every digit from 0 to 9. During training, the weights and biases are updated after every image using gradient descent.

This is an educational implementation, not a production classifier. Its small architecture, short training run, and use of sigmoid outputs with mean-squared error keep the code approachable, but they also limit accuracy. If you are experimenting, useful next steps are increasing the number of epochs, adding a test-set evaluation, or replacing the output/loss pair with softmax and cross-entropy.

## License

No license has been specified for this repository yet. If you plan to reuse or distribute the code, add a license file that matches your intended terms.

