import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Retrieve image arrays and labels from the specified directory
    images, labels = load_data(sys.argv[1])

    # Split the dataset into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Obtain a compiled neural network
    model = get_model()

    # Train the model using the training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Assess the model's performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save the trained model to a file if a filename is provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    """
    Load images and their corresponding labels from the given directory.

    Each subdirectory in `data_dir` should be named with a numerical label,
    and it contains the image files for that label.

    Returns a tuple `(images, labels)`. `images` is a list of
    numpy ndarrays representing the images, and `labels` is a list of
    numerical labels corresponding to those images.
    """
    print(f'Loading images from dataset in directory "{data_dir}"')

    images = []
    labels = []

    # Iterate through category folders in the specified directory
    for foldername in os.listdir(data_dir):
        # Validate that folder name is an integer
        try:
            int(foldername)
        except ValueError:
            print("Warning! Non-integer folder name detected in data directory! Skipping...")
            continue

        # Iterate through images in each category folder
        for filename in os.listdir(os.path.join(data_dir, foldername)):
            img_path = os.path.join(data_dir, foldername, filename)
            img = cv2.imread(img_path)
            if img is not None:  # Ensure image was successfully loaded
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(int(foldername))

    # Validate that number of images matches number of labels
    if len(images) != len(labels):
        sys.exit('Error when loading data: number of images does not match number of labels!')
    else:
        print(f'{len(images)}, {len(labels)} labeled images loaded successfully from dataset!')

    return (images, labels)

def get_model():
    """
    Constructs and returns a compiled convolutional neural network model.
    Assumes the input shape for the first layer is (IMG_WIDTH, IMG_HEIGHT, 3).
    The output layer contains `NUM_CATEGORIES` units, one for each category.
    """

    model = tf.keras.models.Sequential([
        # Add two convolutional layers with 64 filters and 3x3 kernels, followed by pooling layers
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Add a dense layer with 512 units and 50% dropout for regularization
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add the output layer with 43 output units
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model with appropriate settings
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    main()
