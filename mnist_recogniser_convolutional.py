import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist

MODEL_PATH = "models/mnist_model.h5"
MODEL_CHECKPOINT = "data/mnsit_model_checkpoint.weights.h5"
DATA_PATH = "data/mnist_cached_data.npz"

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="linear"),
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def prepare_test_data(visualise: bool = True, cache_file: str = DATA_PATH):

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file)
        return (data["X_train"], data["y_train"]), (data["X_test"], data["y_test"])

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if visualise:
        plt.figure(figsize=(10, 10))
        for i in range(25):  # Display the first 25 images
            plt.subplot(5, 5, i + 1)
            plt.imshow(X_train[i], cmap="gray")  # Use grayscale for MNIST images
            plt.axis("off")
            plt.title(f"Label: {y_train[i]}")

        plt.tight_layout()
        plt.show()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    np.savez(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    return (X_train, y_train), (X_test, y_test) 


def test_model(display_errors: bool = True):
   # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        print("No trained model found! Please train the model first.")
        return

    # Load the model
    model = load_model(MODEL_PATH)

    # Load the test data
    (_, _), (X_test, y_test) = prepare_test_data(visualise=False)

    # Evaluate the model on the test dataset
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    if display_errors:
        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Convert predictions to class labels (taking the argmax)
        predicted_classes = np.argmax(predictions, axis=1)
        # true_classes = np.argmax(y_test, axis=1)  # If y_test is one-hot encoded, convert to class labels

        # Find the indices of the incorrect predictions
        incorrect_indices = np.where(predicted_classes != y_test)[0]

        # Display a few examples of incorrect predictions
        num_display = 25  # Number of incorrect predictions to display
        plt.figure(figsize=(10, 10))
        for i in range(min(num_display, len(incorrect_indices))):
            idx = incorrect_indices[i]
            plt.subplot(5, 5, i + 1)
            plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")  # Reshape back to 28x28 for display
            plt.title(f"True: {y_test[idx]}, Pred: {predicted_classes[idx]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()


# Function to train the model
def train_model():
    # Load dataset
    (X_train, y_train), _ = prepare_test_data(visualise=False)
    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT, save_best_only=True, save_weights_only=True)

    # Create and train model
    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# Function to load the model and make predictions
def predict(image):
    if not os.path.exists(MODEL_PATH):
        print("No trained model found! Please train the model first.")
        return

    # Load the model
    model = load_model(MODEL_PATH)

    # Preprocess the input image
    image = image.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Make predictions
    prediction = model.predict(image)
    print(f"Predicted class: {prediction.argmax()}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare_data", "train", "predict", "test_model"], required=True, help="Mode: 'train' to train the model, 'predict' to use the model.")
    parser.add_argument("--image", type=str, help="Path to the image for prediction (required for 'predict' mode).")

    args = parser.parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        if args.image is None:
            print("Please provide an image path using --image when in 'predict' mode.")
        else:
            # Load and preprocess the image (example assumes grayscale 28x28 image)
            image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not load image from path: {args.image}")
            else:
                predict(image)
    elif args.mode == "prepare_data":
        prepare_test_data()
    elif args.mode == "test_model":
        test_model()
