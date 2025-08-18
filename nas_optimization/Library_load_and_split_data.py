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
    # Modify the paths below to point to the locations of your .idx3 and .idx1 files
    # Example:
    # X = read_idx3(r'C:\path\to\your\all_sessions_pcap_duplicate.idx3')
    # y = read_idx1(r'C:\path\to\your\all_labels_pcap_duplicate.idx1')

    X = read_idx3('/home/adel99/Documents/idx/all_sessions_pcap_duplicate.idx3')
    y = read_idx1('/home/adel99/Documents/idx/all_labels_pcap_duplicate.idx1')

    # Splitting off the test set before preprocessing
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Preprocess training/validation and test data without converting labels
    X_train_val_processed, y_train_val_processed = preprocess_data(X_train_val, y_train_val, num_classes, encode_labels)
    X_test_processed, y_test_processed = preprocess_data(X_test, y_test, num_classes, encode_labels)

    return X_train_val_processed, y_train_val_processed, X_test_processed, y_test_processed

def preprocess_data(X, y, num_classes=11, encode_labels=False):
    X = X.reshape((X.shape[0], -1, 1)) / 255.0
    if encode_labels:
        y = to_categorical(y, num_classes)
    return X, y



num_classes = 11
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

# Now folds are generated and can be used for training/validation
folds = fold_index(X_train_val, y_train_val, num_splits=2)