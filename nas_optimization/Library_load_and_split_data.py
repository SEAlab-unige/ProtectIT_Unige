import numpy as np
import struct
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold

# IDX file reading functions
def read_idx3(filename):
    with open(filename, 'rb') as file:
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError("Invalid IDX3 file format")
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows * cols)  # Flatten images
        return images


def read_idx1(filename):
    with open(filename, 'rb') as file:
        magic, num_items = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError("Invalid IDX1 file format")
        labels = np.fromfile(file, dtype=np.uint8)
        return labels


def load_and_preprocess_data(num_classes, test_size=0.1, encode_labels=False):
    # === USER NOTE: Change these paths to point to your own IDX files ===
    # strategy2.idx3: Feature data (like MNIST images or traffic matrices)
    # strategy2.idx1: Label data (integer class labels)
    #
    # Example: place files in a 'data/' folder and update paths accordingly:
    # X = read_idx3('./data/strategy2.idx3')
    # y = read_idx1('./data/strategy2.idx1')

    X = read_idx3(r'/path/to/your/sessions_data.idx3') # <-- Change this path
    y = read_idx1(r'/path/to/your/labels_data.idx1') # <-- Change this path

    # Splitting off the test set before preprocessing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Preprocess training/validation and test data without converting labels
    X_train_val_processed, y_train_val_processed = preprocess_data(X_train_val, y_train_val, num_classes, encode_labels)
    X_test_processed, y_test_processed = preprocess_data(X_test, y_test, num_classes, encode_labels)

    return X_train_val_processed, y_train_val_processed, X_test_processed, y_test_processed

def preprocess_data(X, y, num_classes=11, encode_labels=False):
    """
    Normalizes inputs to [0,1], reshapes to (samples, 784, 1), and optionally one-hot encodes labels.
    Suitable for Conv1D input.
    """
    X = X.reshape((X.shape[0], -1, 1)) / 255.0
    if encode_labels:
        y = to_categorical(y, num_classes)
    return X, y


# Example usage with the inclusion of a test set
num_classes = 11  # Based on the dataset
X_train_val, y_train_val, X_test, y_test = load_and_preprocess_data(num_classes)

# KFold configuration
def fold_index(X, y, num_splits=2):
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    return list(skf.split(X, y))


# Getting the fold splits
def get_fold_split(X, Y, folds, i_fold):
    train_ind, test_ind = folds[i_fold]
    X_train = X[train_ind]
    Y_train = Y[train_ind]
    X_val = X[test_ind]
    Y_val = Y[test_ind]
    return X_train, Y_train, X_val, Y_val

# Ensure folds are correctly generated and can be used for training/validation
folds = fold_index(X_train_val, y_train_val, num_splits=5)
folds = fold_index(X_train_val, y_train_val, num_splits=2)
