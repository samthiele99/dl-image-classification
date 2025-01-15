import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


MODEL_PATH = "models/fashion_mnist_model.h5"
MODEL_CHECKPOINT = "data/fashion_mnsit_model_checkpoint.weights.h5"
DATA_PATH = "data/fashion_mnist_cached_data.npz"
BASE_LEARNING_RATE = 0.001

# Got cross validation of 91 % with current set up.

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.5),
        Dense(10, activation="linear"),
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


def prepare_test_data(visualise: bool = True, cache_file: str = DATA_PATH):

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file)
        return (data["X_train"], data["y_train"]), (data["X_test"], data["y_test"]), (data["X_cv"], data["y_cv"])

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

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
    X_cv = X_cv.astype("float32") / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_cv = X_cv.reshape(-1, 28, 28, 1)
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    np.savez(cache_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_cv=X_cv, y_cv=y_cv)

    return (X_train, y_train), (X_test, y_test), (X_cv, y_cv)


def test_model(display_errors: bool = True, cross_validate: bool = True):
   # Check if the model exists
    if not os.path.exists(MODEL_PATH):
        print("No trained model found! Please train the model first.")
        return

    # Load the model
    model = load_model(MODEL_PATH)

    # Load the test data
    (_, _), (X_test, y_test), (X_cv, y_cv) = prepare_test_data(visualise=False)

    # Evaluate the model on the test dataset
    if not cross_validate:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    else:
        test_loss, test_accuracy = model.evaluate(X_cv, y_cv, verbose=2)
        print(f"Cross Validation Loss: {test_loss}")
        print(f"Cross Validation Accuracy: {test_accuracy * 100:.2f}%")

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
    (X_train, y_train), _, _ = prepare_test_data(visualise=False)
    # Maually split the training set into validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Checkpoint that is saved as the model runs
    checkpoint = ModelCheckpoint(MODEL_CHECKPOINT, save_best_only=True, save_weights_only=True)

    # Generator the shift image for each batch
    datagen = ImageDataGenerator(
        rotation_range=10,        # Rotate images by up to 10 degrees
        width_shift_range=0.1,    # Shift width by 10%
        height_shift_range=0.1,   # Shift height by 10%
        horizontal_flip=False,    # Horizontal flip (not used for Fashion MNIST)
    )

    # Fit the generator to the training data
    datagen.fit(X_train)

    # Create and train model
    model = create_model()
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: BASE_LEARNING_RATE * 0.1**(epoch // 10))
    model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, lr_scheduler]
    )

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
    parser.add_argument("--cross-validate", action="store_true", 
                    help="If included, performs cross-validation during training.")

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
        test_model(cross_validate=args.cross_validate)
