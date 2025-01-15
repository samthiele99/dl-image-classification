import argparse
import os
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

MODEL_PATH = 'my_model.h5'

# Function to create the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy'],
    )
    return model

# Function to train the model
def train_model():
    # Load dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create and train model
    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

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
    image = image.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Make predictions
    prediction = model.predict(image)
    print(f"Predicted class: {prediction.argmax()}")

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help="Mode: 'train' to train the model, 'predict' to use the model.")
    parser.add_argument('--image', type=str, help="Path to the image for prediction (required for 'predict' mode).")

    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if args.image is None:
            print("Please provide an image path using --image when in 'predict' mode.")
        else:
            # Load and preprocess the image (example assumes grayscale 28x28 image)
            import cv2
            image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not load image from path: {args.image}")
            else:
                predict(image)
