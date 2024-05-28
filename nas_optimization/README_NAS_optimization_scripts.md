# NAS Optimization Scripts README

This repository contains scripts for Neural Architecture Search (NAS) optimization. The primary functionalities include defining network architectures, performing NAS, computing performance statistics, and handling data loading and splitting.

## Scripts Overview

### `B01_NAS.py`
The main script that orchestrates the NAS process. It sets up the data, initializes NAS with hardware constraints, and runs the NAS process using specified parameters.

- **Key Details:**
  - **Flags for Training:**
    - `is_train`: Indicates whether to perform training. If `True`, training is performed.
    - `is_train_proxy`: Indicates whether to perform proxy training. If `True`, proxy training is performed.
    - `use_full_training`: Determines the training routine to use. If `True`, uses `train_routine`. If `False`, uses `proxy_train_routine` for faster training.
  - **Data Loading and Preprocessing:**
    - Loads and preprocesses data without encoding labels for stratified folds generation.
  - **Stratified K-Fold Generation:**
    - Generates stratified K-Fold indices for cross-validation.
  - **Label Conversion:**
    - Converts labels to one-hot encoding after creating folds.
  - **Preloaded Data:**
    - Packs the preloaded data for NAS.
  - **Timestamp Generation:**
    - Creates a unique timestamp for this run.
  - **NAS Initialization:**
    - Initializes the NAS with specified parameters, including hardware constraints:
      - `max_depth_father`: Maximum depth of the parent network.
      - `max_depth`: Maximum depth of the child networks.
      - `check_hw`: Enables hardware checks.
      - `params_thr`: Threshold for the number of parameters.
      - `flops_thr`: Threshold for the number of FLOPs.
      - `flash_thr`: Threshold for flash memory usage.
      - `ram_thr`: Threshold for RAM usage.
      - `n_generations`: Number of generations to run.
      - `n_child`: Number of child networks per generation.
      - `n_mutations`: Number of mutations per network.
      - `partial_save_steps`: Interval for saving partial results.
      - `smart_start`: Enables smart initialization with predefined blocks.
      - `is_random_walk`: Enables random walk mutations.

### `library_net.py`
Defines the network architectures and the training routines.
- **Key Functions:**
  - `__init__(self, block_list, nas_saver_name, preloaded_data=None)`: Initializes the network with a list of blocks and optional preloaded data.
  - `fetch_data(self, num_classes=11, test_size=0.1, encode_labels=False)`: Loads and preprocesses data if not already provided.
  - `short_description(self)`: Logs a brief description of the network, including hardware parameters.
  - `dump(self)`: Logs detailed information about each block in the network.
  - `ins_keras_model(self, load_weigths=False)`: Constructs the Keras model from defined blocks, optionally loading pretrained weights.
  - `train_routine(self, is_train, folds)`: Handles the training of the network using K-Fold cross-validation.
  - `proxy_train_routine(self, is_train_proxy, folds, num_selected_folds=2)`: A faster training routine using a subset of folds for proxy training.
  - `hw_measures(self)`: Computes and returns hardware-related measures such as number of parameters, max tensor size, FLOPs, flash size, and RAM size.
  - `log_message(self, message)`: Logs messages to a specified NAS log file.

### `library_nas.py`
Implements the NAS algorithms, including mutations and generation management.
- **Key Functions:**
  - `run_NAS(self, folds)`: Executes the NAS optimization process across multiple generations.
  - `train_generation(self, child_set, folds)`: Trains a generation of child networks using full or proxy training.
  - `best_child_selection(self, child_set, gen_per, test_metrics_list)`: Selects the best performing child network based on validation metrics.
  - `one_nas_step(self, parent, folds)`: Executes one step of NAS, including generation of children, training, and selection.
  - `new_generation(self, parent, preloaded_data)`: Generates a new set of child networks through mutation.
  - `mutate_network_and_control(self, parent, preloaded_data)`: Mutates a parent network while checking hardware constraints.
  - `mutate_network(self, parent, preloaded_data)`: Performs mutation actions (remove, add, or change a block) on the parent network.
  - `generate_random_block(self, input_size, block_index, current_architecture_depth)`: Generates a random block with specified parameters.
  - `remove_block(self, parent_blocks)`: Removes a block from the network.
  - `change_block(self, parent_blocks)`: Changes a block in the network.
  - `add_block(self, parent_blocks)`: Adds a new block to the network.
  - `recalculate_dropout_rates(self, blocks)`: Recalculates dropout rates for the blocks in the network.
  - `correct_blocklist(self, child_blocks)`: Corrects the block list to ensure valid configurations.
  - `save_partial(self, parent, best_network, i_gen, gen_best_score, absolute_best_score)`: Saves the parent and best network models at a given generation.
  - `log_message(self, message, mode='a')`: Logs messages to a specified NAS log file.


### `library_block.py`
Defines the building blocks used to construct the network architectures.
- **Key Functions:**
  - `calculate_output_size(self)`: Calculates the output size of the block based on input size, stride, and pooling.
  - `dump(self)`: Dumps the block configuration to a log file in JSON format.
  - `create_layer(self, input_shape=None)`: Generates a list of Keras layers based on the block configuration.


### `library_compute_stats.py`
Computes various statistics related to network performance.
- **Key Functions:**
  - `compute_descriptors(y_true, y_pred)`: Calculates accuracy, precision, recall, and F1-score for the model predictions.


### `library_load_and_split_data.py`
Handles data loading, preprocessing, and splitting into training, validation, and test sets.
- **Key Functions:**
  - `read_idx3(filename)`: Reads and returns session data from an IDX3 file.
  - `read_idx1(filename)`: Reads and returns label data from an IDX1 file.
  - `load_and_preprocess_data(num_classes, test_size=0.1, encode_labels=False)`: Loads, preprocesses, and splits data into training, validation, and test sets.
  - `preprocess_data(X, y, num_classes=11, encode_labels=False)`: Normalizes the data and optionally encodes labels.
  - `fold_index(X, y, num_splits=2)`: Generates stratified K-Fold indices for cross-validation.
  - `get_fold_split(X, Y, folds, i_fold)`: Retrieves training and validation splits for a specific fold.

### Dependencies
- Python 3
- TensorFlow
- Keras
- Scikit-learn
- Numpy
- Keras-flops
- Json
- Gc
- Random
- Os

