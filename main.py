
import numpy as np

import utils



RNG = np.random.default_rng(42)
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
CLASSES = 10
EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.08


def softmax(logits):
    """Numerically stable softmax for a batch of rows."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def initialize_weights():
    """Create He-initialized weights for the ReLU hidden layer."""
    weights_input_to_hidden = RNG.normal(
        0, np.sqrt(2 / INPUT_SIZE), size=(INPUT_SIZE, HIDDEN_SIZE)
    ).astype(np.float32)
    weights_hidden_to_output = RNG.normal(
        0, np.sqrt(2 / HIDDEN_SIZE), size=(HIDDEN_SIZE, CLASSES)
    ).astype(np.float32)
    return (
        weights_input_to_hidden,
        np.zeros(HIDDEN_SIZE, dtype=np.float32),
        weights_hidden_to_output,
        np.zeros(CLASSES, dtype=np.float32),
    )


def forward(images, weights_input_to_hidden, bias_input_to_hidden,
            weights_hidden_to_output, bias_hidden_to_output):
    hidden_raw = images @ weights_input_to_hidden + bias_input_to_hidden
    hidden = np.maximum(hidden_raw, 0)  # ReLU
    probabilities = softmax(hidden @ weights_hidden_to_output + bias_hidden_to_output)
    return hidden_raw, hidden, probabilities


def accuracy(images, labels, weights_input_to_hidden, bias_input_to_hidden,
             weights_hidden_to_output, bias_hidden_to_output):
    _, _, probabilities = forward(
        images, weights_input_to_hidden, bias_input_to_hidden,
        weights_hidden_to_output, bias_hidden_to_output,
    )
    return np.mean(probabilities.argmax(axis=1) == labels.argmax(axis=1))


def train(images, labels):
    model = initialize_weights()
    (weights_input_to_hidden, bias_input_to_hidden,
     weights_hidden_to_output, bias_hidden_to_output) = model

    for epoch in range(1, EPOCHS + 1):
        order = RNG.permutation(len(images))
        total_loss = 0.0

        for start in range(0, len(images), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_images = images[indices]
            batch_labels = labels[indices]
            batch_size = len(batch_images)

            hidden_raw, hidden, probabilities = forward(
                batch_images, *model
            )
            total_loss += -np.sum(batch_labels * np.log(probabilities + 1e-8))

            # Back-propagate all gradients before changing any weights.
            delta_output = (probabilities - batch_labels) / batch_size
            grad_hidden_to_output = hidden.T @ delta_output
            grad_bias_hidden_to_output = delta_output.sum(axis=0)
            delta_hidden = (
                delta_output @ weights_hidden_to_output.T
            ) * (hidden_raw > 0)
            grad_input_to_hidden = batch_images.T @ delta_hidden
            grad_bias_input_to_hidden = delta_hidden.sum(axis=0)

            weights_hidden_to_output -= LEARNING_RATE * grad_hidden_to_output
            bias_hidden_to_output -= LEARNING_RATE * grad_bias_hidden_to_output
            weights_input_to_hidden -= LEARNING_RATE * grad_input_to_hidden
            bias_input_to_hidden -= LEARNING_RATE * grad_bias_input_to_hidden

        train_accuracy = accuracy(images, labels, *model)
        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"loss={total_loss / len(images):.4f}, accuracy={train_accuracy:.2%}"
        )

    return model


def predict_image(image_path, model):
    image = utils.prepare_digit(image_path).reshape(1, -1)
    _, _, probabilities = forward(image, *model)
    return int(probabilities.argmax()), float(probabilities.max() * 100)


def recognize_images(model, images_dir="images"):
    image_files = utils.image_paths(images_dir)
    if not image_files:
        print(f"No images were found in '{images_dir}'.")
        return

    for image_path in image_files:
        try:
            digit, confidence = predict_image(image_path, model)
            print(f"{image_path.name}: digit {digit}, confidence {confidence:.1f}%")
        except ValueError as error:
            print(f"{image_path.name}: skipped — {error}")


if __name__ == "__main__":
    images, labels = utils.load_dataset()
    recognize_images(train(images, labels))
