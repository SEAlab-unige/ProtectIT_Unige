import tensorflow as tf
import Library_load_and_split_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from Library_load_and_split_data import get_fold_split
import gc
from sklearn.metrics import accuracy_score
import numpy as np
import Library_compute_stats
from keras_flops import get_flops
from Library_Block import Block
from sklearn.model_selection import StratifiedKFold

class Net:
    # Initialize the network with a list of blocks and optional preloaded data.
    # If data is not preloaded, it must be loaded later via fetch_data().
    def __init__(self, block_list, nas_saver_name, preloaded_data=None):
        self.block_list = block_list
        self.nas_saver_name = nas_saver_name
        self.trained_fully = False
        # Directly use the preloaded data
        if preloaded_data:
            self.X_train_val, self.y_train_val, self.X_test, self.y_test = preloaded_data
        else:
            self.X_train_val = self.y_train_val = self.X_test = self.y_test = None

    # Load and split the dataset if not already loaded.
    def fetch_data(self, num_classes=11, test_size=0.1, encode_labels=False):
        # If data has not already been set, load it
        if self.X_train_val is None or self.X_test is None:
            print("Loading and preprocessing data...")
            X_train_val, y_train_val, X_test, y_test = Library_load_and_split_data.load_and_preprocess_data(num_classes, test_size, encode_labels=encode_labels)
            self.X_train_val, self.y_train_val = X_train_val, y_train_val
            self.X_test, self.y_test = X_test, y_test
        else:
            print("Using preloaded data.")

        return self.X_train_val, self.y_train_val, self.X_test, self.y_test


    ## Log a compact summary of network structure and hardware measures
    def short_description(self):
        hw_params = self.hw_measures()
        self.log_message(
            f"Net: len = {len(self.block_list)}, n_params = {hw_params[0]}, max_tens = {hw_params[1]}, "
            f"flops = {hw_params[2]}, flash_size = {hw_params[3]} bytes, ram_size = {hw_params[4]} bytes"
        )
        return True

    # Call dump() for each block to log architecture details
    def dump(self):
        self.log_message(' NET: ')
        for i in range(len(self.block_list)):
            self.block_list[i].dump()

    # Construct a Keras Sequential model from the block list.
    def ins_keras_model(self, load_weigths=False):
        model = tf.keras.models.Sequential()

        for i, block in enumerate(self.block_list):  # Use self.block_list directly
            if i == 0:
                # For the first block, set input_shape explicitly and avoid BatchNormalization
                keras_layers = block.create_layer(input_shape=(784, 1), is_first_layer=True)
            else:
                keras_layers = block.create_layer(is_first_layer=False)

            for layer in keras_layers:
                model.add(layer)

        # GlobalAveragePooling, Flatten, and Dense layers remain the same
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Flatten())
        num_classes = 11  # Adjust based on your number of classes
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        if load_weigths:
            for i in range(0, len(model.weights) - 2, 2):
                if self.block_list[int(i / 2)].has_trained_weigths:
                    model.weights[i] = self.block_list[int(i / 2)].trained_weights[0]
                    model.weights[i + 1] = self.block_list[int(i / 2)].trained_weights[1]
            if self.trained_fully is not None:
                model.weights[-2] = self.trained_fully[0]
                model.weights[-1] = self.trained_fully[1]

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Run full K-fold training and evaluation using early stopping and learning rate scheduling.
    # Returns validation scores and average test metrics per fold.
    def train_routine(self, is_train, folds):
        learning_rate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)
        n_epochs = 60
        multistart = 1
        batch_size = 64

        self.fetch_data(num_classes=11, test_size=0.1, encode_labels=False)
        y_test_c = self.y_test  # Assuming y_test is already one-hot encoded

        results = []
        for i_fold in range(len(folds)):
            X_train, y_train, X_val, y_val = Library_load_and_split_data.get_fold_split(
                self.X_train_val, self.y_train_val, folds, i_fold)

            all_test_metrics = np.zeros(5)  # Extend to capture the average of accuracy, precision, recall, and F1-score

            for i_mult in range(multistart):
                model = self.ins_keras_model()
                if is_train:
                    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                              callbacks=[learning_rate_cb, early_stop_cb])

                p_val = model.predict(X_val)
                metrics = np.array(Library_compute_stats.compute_descriptors(y_val, p_val))
                val_score = np.mean(metrics[:4])  # Exclude average from validation score

                p_test = model.predict(self.X_test)
                test_metrics = np.array(Library_compute_stats.compute_descriptors(y_test_c, p_test))
                test_metrics = np.append(test_metrics, np.mean(test_metrics[:4]))  # Append the average of first four test metrics  # Append the average of test metrics
                all_test_metrics += test_metrics

                del model
                gc.collect()
                tf.keras.backend.clear_session()

            avg_test_metrics = all_test_metrics / multistart
            results.append((val_score, avg_test_metrics.tolist()))
            print(f"Fold {i_fold + 1} - Validation Metrics: Avg Score={val_score}")
            print(
                f"Fold {i_fold + 1} - Average Test Metrics: Accuracy={avg_test_metrics[0]}, Precision={avg_test_metrics[1]}, Recall={avg_test_metrics[2]}, F1={avg_test_metrics[3]}, Average={avg_test_metrics[4]}")

        return results

    # Run a faster proxy training using a single fold and internal stratified validation split.
    # Returns best validation and test accuracy from multiple restarts.
    def proxy_train_routine(self, is_train_proxy, selected_fold_index=0, validation_split=0.11, folds=None):
        learning_rate_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=12)
        early_stop_cb = EarlyStopping(monitor='val_loss', patience=20)
        n_epochs = 200
        multistart = 1
        batch_size = 256

        self.fetch_data(num_classes=11, test_size=0.1, encode_labels=False)
        y_test_c = self.y_test

        best_val_accs = []
        best_test_accs = []

        print(f"Selected fold index: {selected_fold_index}")
        X_train_fold, y_train_fold, X_val_fold, y_val_fold = get_fold_split(self.X_train_val, self.y_train_val, folds,
                                                                            selected_fold_index)

        # Convert one-hot encoded labels to single integer labels
        y_train_fold_single = np.argmax(y_train_fold, axis=1)

        # Stratified splitting for internal training and validation within the selected fold
        stratified_split = StratifiedKFold(n_splits=int(1 / validation_split), shuffle=True, random_state=42)
        for train_idx, val_idx in stratified_split.split(X_train_fold, y_train_fold_single):
            X_train, X_val = X_train_fold[train_idx], X_train_fold[val_idx]
            y_train, y_val = y_train_fold[train_idx], y_train_fold[val_idx]
            break  # Only take the first split

        best_val_acc = 0
        best_test_acc = None

        for i_mult in range(multistart):
            model = self.ins_keras_model()
            if is_train_proxy:
                model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size,
                          validation_data=(X_val, y_val),
                          callbacks=[learning_rate_cb, early_stop_cb])

            p_val = model.predict(X_val)
            predicted_labels = np.argmax(p_val, axis=1)
            true_labels = np.argmax(y_val, axis=1)

            val_acc = accuracy_score(true_labels, predicted_labels)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                p_test = model.predict(self.X_test)
                test_predicted_labels = np.argmax(p_test, axis=1)
                test_true_labels = np.argmax(y_test_c, axis=1)
                test_acc = accuracy_score(test_true_labels, test_predicted_labels)
                best_test_acc = test_acc

            del model
            gc.collect()
            tf.keras.backend.clear_session()

        best_val_accs.append(best_val_acc)
        if best_test_acc is not None:
            best_test_accs.append(best_test_acc)

        avg_best_val_acc = sum(best_val_accs) / len(best_val_accs) if best_val_accs else 0
        avg_best_test_acc = sum(best_test_accs) / len(best_test_accs) if best_test_accs else 0

        print(f"Average Best validation accuracy: {avg_best_val_acc}")
        print(f"Average Best test accuracy: {avg_best_test_acc}")

        return [(avg_best_val_acc, avg_best_test_acc)]

    # Compute and return hardware-related metrics: parameter count, max tensor size, FLOPs,
    def hw_measures(self):
        model = self.ins_keras_model()
        n_params = model.count_params()

        # Maximum tensor size propagated
        max_tens = max(
            [np.prod(layer.output_shape[1:]) for layer in model.layers if None not in layer.output_shape[1:]],
            default=0  # Default to 0 if list comprehension results in an empty list
        )

        # Calculate FLOPs using keras_flops
        try:
            flops = get_flops(model, batch_size=1)
        except Exception as e:
            print(f"Error calculating FLOPs: {e}")
            flops = 0  # Defaulting to 0 if FLOPs calculation fails

        flash_size = 4 * n_params  # Assuming 4 bytes per parameter
        ram_size = 4 * max_tens  # Assuming 4 bytes per element in the largest tensor

        print(f"FLOPs: {flops}")  # Printing for easy debugging
        return [n_params, max_tens, flops, flash_size, ram_size]

    # Append a string message to the NAS log file for experiment tracking.
    def log_message(self, message):
        """Logs a message to the specified NAS log file."""
        with open(self.nas_saver_name + '.txt', 'a') as log_file:
            log_file.write(message + '\n')
