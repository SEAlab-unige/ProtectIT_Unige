import tensorflow as tf
import Library_load_and_split_data
from sklearn.model_selection import train_test_split
import gc
from sklearn.metrics import accuracy_score
import numpy as np
import Library_compute_stats
from keras_flops import get_flops
from Library_Block import Block


class Net:
    def __init__(self, block_list, nas_saver_name, preloaded_data=None):
        self.block_list = block_list
        self.nas_saver_name = nas_saver_name
        self.trained_fully = False
        # Directly use the preloaded data
        if preloaded_data:
            self.X_train_val, self.y_train_val, self.X_test, self.y_test = preloaded_data
        else:
            self.X_train_val = self.y_train_val = self.X_test = self.y_test = None


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



    def short_description(self):
        hw_params = self.hw_measures()
        self.log_message(
            f"Net: len = {len(self.block_list)}, n_params = {hw_params[0]}, max_tens = {hw_params[1]}, "
            f"flops = {hw_params[2]}, flash_size = {hw_params[3]} bytes, ram_size = {hw_params[4]} bytes"
        )
        return True

    def dump(self):
        self.log_message(' NET: ')
        for i in range(len(self.block_list)):
            self.block_list[i].dump()

    def ins_keras_model(self, load_weigths = False):
        model = tf.keras.models.Sequential()

        for i, block in enumerate(self.block_list):  # # Use self.block_list directly
            if i == 0:
                # For the first block, set input_shape explicitly
                keras_layers = block.create_layer(input_shape=(784, 1))
            else:
                keras_layers = block.create_layer()

            for layer in keras_layers:
                model.add(layer)

        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Flatten())
        num_classes = 11
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        if(load_weigths==True):
            for i in range(0, len(model.weights)-2, 2):
                if(self.block_list[int(i/2)].has_trained_weigths== True):
                    model.weights[i] = self.block_list[int(i/2)].trained_weights[0]
                    model.weights[i+1] = self.block_list[int(i/2)].trained_weights[1]
            if(self.trained_fully != None):
                model.weights[len(model.weights)-2] =self.trained_fully[0]
                model.weights[len(model.weights)-1] =self.trained_fully[1]

        #model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_routine(self, is_train, folds):
        learning_rate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)
        n_epochs = 100
        multistart = 4
        batch_size = 128

        self.fetch_data(num_classes=11, test_size=0.1, encode_labels=False)
        y_test_c = self.y_test  # y_test is already one-hot encoded

        results = []
        for i_fold in range(len(folds)):
            X_train, y_train, X_val, y_val = Library_load_and_split_data.get_fold_split(
                self.X_train_val, self.y_train_val, folds, i_fold)

            all_test_metrics = np.zeros(5)  # this list is to capture the average of accuracy, precision, recall, and F1-score

            for i_mult in range(multistart):
                model = self.ins_keras_model()
                if is_train:
                    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                              callbacks=[learning_rate_cb, early_stop_cb])

                p_val = model.predict(X_val)
                metrics = np.array(Library_compute_stats.compute_descriptors(y_val, p_val))
                val_score = np.mean(metrics[:4])

                p_test = model.predict(self.X_test)
                test_metrics = np.array(Library_compute_stats.compute_descriptors(y_test_c, p_test))
                test_metrics = np.append(test_metrics, np.mean(test_metrics[:4]))  # Append the average of first four test metrics
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

    def proxy_train_routine(self, is_train_proxy, folds, num_selected_folds=2):
        learning_rate_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
        n_epochs = 100
        multistart = 2
        batch_size = 256

        self.fetch_data(num_classes=11, test_size=0.1, encode_labels=False)
        y_test_c = self.y_test

        selected_folds = np.random.choice(len(folds), size=num_selected_folds, replace=False)

        best_val_accs = []
        best_test_accs = []

        for i_fold in selected_folds:
            print(f"Selected fold index: {i_fold} out of {len(folds)} folds")
            X_train, y_train, X_val, y_val = Library_load_and_split_data.get_fold_split(
                self.X_train_val, self.y_train_val, folds, i_fold)

            best_val_acc = 0
            best_test_acc = None

            for i_mult in range(multistart):
                model = self.ins_keras_model()
                if is_train_proxy:
                    model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
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

    def log_message(self, message):
        """Logs a message to the specified NAS log file."""
        with open(self.nas_saver_name + '.txt', 'a') as log_file:
            log_file.write(message + '\n')