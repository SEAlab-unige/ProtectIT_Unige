import json
from tensorflow.keras import layers
import tensorflow as tf

class Block:
    """
        Represents a modular block of layers in a CNN architecture for NAS.
        Combines Conv1D, optional pooling, activation, dropout, and batch normalization.
    """

    def __init__(self, n_filters=None, kernel_size=None, activation="relu", padding="valid",
                 is_pool=False, pool_size=2, input_size=None, is_dropout=False, dropout_rate=0.5,
                 stride=2, nas_saver_name="NAS_logger", input_shape=None, is_max_pool=False, is_avg_pool=False):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding
        self.is_pool = is_pool
        self.pool_size = pool_size
        self.input_size = input_size
        self.is_dropout = is_dropout
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.nas_saver_name = nas_saver_name
        self.input_shape = input_shape
        self.is_max_pool = is_max_pool
        self.is_avg_pool = is_avg_pool

        # Ensure that either is_max_pool or is_avg_pool is True when is_pool is True
        if self.is_pool and not (self.is_max_pool or self.is_avg_pool):
            self.is_max_pool = True  # Default to max pooling if none are set

        self.output_size = self.calculate_output_size()

    def calculate_output_size(self):
        """
        Calculate the output size of the block considering both convolution and pooling operations.
        Adjusts invalid configurations dynamically (e.g., switches from "valid" to "same" padding).
        """

        if self.input_size is None or self.kernel_size is None:
            raise ValueError("input_size and kernel_size must be provided to calculate output size.")

        # Strict constraint: Avoid "valid" padding if input size is less than 32
        if self.input_size < 32 and self.padding == "valid":
            self.padding = "same"

        # Adjust kernel size if it exceeds input size
        if self.kernel_size > self.input_size:
            self.kernel_size = self.input_size

        # Enforce valid padding type
        if self.padding not in ["same", "valid"]:
            self.padding = "same"  # Default to "same" padding if invalid

        # Switch to "same" padding if "valid" is infeasible
        if self.padding == "valid" and self.input_size < self.kernel_size:
            self.padding = "same"

        # Convolution output size calculation
        if self.padding == "same":
            output_size = (self.input_size + self.stride - 1) // self.stride
        elif self.padding == "valid":
            output_size = (self.input_size - self.kernel_size + self.stride) // self.stride

        # Pooling output size calculation (if applicable)
        if self.is_pool:
            if output_size < self.pool_size:
                # Disable pooling if output_size < pool_size
                self.is_pool = False
                self.is_max_pool = False
                self.is_avg_pool = False
            else:
                output_size = (output_size + self.pool_size - 1) // self.pool_size

        # Ensure output size is valid
        if output_size <= 0:
            # Automatically adjust parameters to avoid invalid configurations
            self.padding = "same"  # Use "same" padding
            self.kernel_size = min(self.kernel_size, self.input_size)  # Adjust kernel size
            self.stride = max(1, self.stride)  # Ensure stride is at least 1
            output_size = (self.input_size + self.stride - 1) // self.stride

        return output_size

    def dump(self):
        config_str = json.dumps(self.__dict__, indent=4)
        with open(f'{self.nas_saver_name}.txt', 'a') as nas_logger:
            nas_logger.write(config_str + "\n")

    """
    Constructs a list of Keras layers from the block configuration.
    Adds Conv1D, optional pooling, activation, dropout, and batch norm.
    """

    def create_layer(self, input_shape=None, is_first_layer=False):
        layers_list = []

        # Convolutional layer
        conv_kwargs = {
            "filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "activation": None,
            "padding": self.padding,
            "strides": self.stride,
        }
        if input_shape:
            conv_kwargs["input_shape"] = input_shape
        layers_list.append(tf.keras.layers.Conv1D(**conv_kwargs))

        # Add BatchNormalization only if it's not the first layer
        if not is_first_layer:
            layers_list.append(tf.keras.layers.BatchNormalization())

        # Activation layer
        if self.activation:
            layers_list.append(tf.keras.layers.Activation(self.activation))

        # Pooling layer (Skip pooling for the first block)
        if self.is_pool and not is_first_layer:
            if self.is_max_pool:
                layers_list.append(tf.keras.layers.MaxPooling1D(pool_size=self.pool_size))
            elif self.is_avg_pool:
                layers_list.append(tf.keras.layers.AveragePooling1D(pool_size=self.pool_size))

        # Dropout layer
        if self.is_dropout:
            layers_list.append(tf.keras.layers.Dropout(self.dropout_rate))

        return layers_list







