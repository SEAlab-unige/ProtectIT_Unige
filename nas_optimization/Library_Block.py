import json
from tensorflow.keras import layers

class Block:
    def __init__(self, n_filters=None, kernel_size=None, activation="relu", padding="same",
                 is_pool=False, pool_size=2, input_size=None, is_dropout=False, dropout_rate=0.5,
                 stride=2, nas_saver_name="NAS_logger", input_shape=None):
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

        # Calculate output_size based on whether pooling is used
        self.output_size = self.calculate_output_size()

    def calculate_output_size(self):
        output_size = self.input_size  # Starting with input_size
        if self.is_pool or self.stride > 1:
            # Adjusting for stride
            output_size = (output_size + self.stride - 1) // self.stride
            # Adjusting for pooling
            if self.is_pool:
                output_size = (output_size + self.pool_size - 1) // self.pool_size
        return output_size

    def dump(self):
        config_str = json.dumps(self.__dict__, indent=4)
        with open(f'{self.nas_saver_name}.txt', 'a') as nas_logger:
            nas_logger.write(config_str + "\n")

    def create_layer(self, input_shape=None):
        """Generates a list of Keras layers based on the block configuration."""
        layers_list = []

        # Convolutional layer
        conv_kwargs = {
            "filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "activation": None,  # Later followed by BatchNormalization and Activation
            "padding": self.padding,
            "strides": self.stride,
        }
        if input_shape:
            conv_kwargs["input_shape"] = input_shape
        layers_list.append(layers.Conv1D(**conv_kwargs))
        # Batch normalization layer
        layers_list.append(layers.BatchNormalization())

        # Activation layer
        if self.activation:
            layers_list.append(layers.Activation(self.activation))

        # Pooling layer
        if self.is_pool:
            layers_list.append(layers.MaxPooling1D(pool_size=self.pool_size))

        # Dropout layer
        if self.is_dropout:
            layers_list.append(layers.Dropout(self.dropout_rate))

        return layers_list

