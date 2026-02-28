# NAS Optimization Scripts

This repository implements a **Neural Architecture Search (NAS)** pipeline with hardware constraints. It is designed to search for optimized deep learning architectures for embedded/edge platforms with strict resource limits.

The core functionalities include:
- Defining and mutating deep learning blocks
- Building and training networks 
- Computing hardware and performance metrics
- Executing NAS optimization over multiple generations

The NAS process is hardware-aware and supports constraints such as parameter count, FLOPs, RAM, and Flash size.

---

## Scripts Overview

### `B01_NAS.py`

This is the **main script** that launches the NAS process. It loads the dataset, sets up hardware-constrained NAS parameters, and runs the search loop.

#### 🔍 Purpose

To initialize and execute a full NAS run, using the defined components in `library_nas.py`, `library_net.py`, and `library_load_and_split_data.py`.

#### 🧠 What It Does

- Loads raw session/traffic data from `.idx3` and `.idx1` files
- Prepares stratified K-Fold cross-validation
- Converts labels to one-hot encoding after fold creation 
- Initializes the NAS engine with user-defined search configuration
- Starts the NAS optimization loop with mutation and selection

#### ⚙️ Key Parameters

- `is_train_proxy`: Enables or disables `.fit()` inside the proxy training routine  
- `is_train`: Enables or disables `.fit()` inside the full training routine
- `use_full_training`: Controls which training routine is used
- `num_classes`: number of output classes (default: 11 for ISCX VPN-nonVPN)
- `max_depth_father`, `max_depth`: Control network architecture depth
- `check_hw`: Enforces hardware constraints
- `params_thr`, `flops_thr`, `flash_thr`, `ram_thr`, `max_tens_thr`: Hardware thresholds
- `n_generations`: Number of generations in the NAS process
- `n_child`: Number of children per generation
- `n_mutations`: Number of mutations applied to each child
- `partial_save_steps`: Interval (in generations) to save progress
- `smart_start`: Initializes with hand-crafted blocks
- `is_random_walk`: Enables stochastic mutations


### `library_net.py`
Defines the **network architecture class** used in NAS, including model construction, training routines, and hardware evaluation.

#### 🔍 Purpose

To:
- Build a sequential Keras model from modular blocks
- Train the model using full or proxy training strategies
- Compute hardware constraints used during NAS search

#### 🔧 Key Functions

  - `__init__(self, block_list, nas_saver_name, preloaded_data=None)`: Initializes the network with a list of blocks and optional preloaded data.
  - `fetch_data(self, num_classes=11, test_size=0.1, encode_labels=False)`: Loads and preprocesses data if not already provided.
  - `short_description(self)`: Logs a brief description of the network, including hardware parameters.
  - `dump(self)`: Logs detailed information about each block in the network.
  - `ins_keras_model(self, load_weigths=False)`: Builds the Keras model from block list. Optionally loads pretrained weights into each block.
  - `train_routine(self, is_train, folds)`: Trains the model using K-Fold cross-validation and computes validation and test metrics.
  - `proxy_train_routine(self, is_train_proxy, folds, num_selected_folds=2)`: Lightweight alternative to full training. Trains using a subset of folds for speed.
  - `hw_measures(self)`: Computes and returns hardware-related measures such as number of parameters, max tensor size, FLOPs, flash size, and RAM size.
  - `log_message(self, message)`: Logs messages to a specified NAS log file.

### `library_nas.py`
This module implements the **NAS engine** that evolves neural networks by applying block-level mutations across multiple generations. It works in conjunction with `library_block.py` (which defines building blocks) and `library_net.py` (which builds and trains networks).

#### 🔍 Purpose

To **automatically discover high-performing network architectures** under hardware constraints by:
- Generating child networks via block-level mutation
- Evaluating candidates with proxy or full training
- **Greedy (hill-climbing-style) selection:** the best-performing *admissible* child becomes the new parent at each generation
- Tracking the **absolute best** model found across all generations
  
#### 🔧 Key Functions
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
  - `correct_blocklist(self, child_blocks)`: Corrects the block list to ensure valid architectural configurations post-mutation.
  - `save_partial(self, parent, best_network, i_gen, gen_best_score, absolute_best_score)`: Saves the parent and best network models at a given generation.
  - `log_message(self, message, mode='a')`: Appends a line to the NAS log file. Used for tracking progress.


### `library_block.py`
Defines modular **building blocks** for constructing convolutional neural networks during NAS.

#### 🔍 Purpose

To represent a reusable layer block that combines convolution, activation, pooling, and dropout.  
Each block is used to build networks dynamically within the NAS framework.

#### 🔧 Key Functions

  - `calculate_output_size(self)`: Computes the block's output size after convolution and pooling, handling invalid settings automatically.
  - `dump(self)`: Logs the block's configuration to a file in readable JSON format for NAS tracking.
  - `create_layer(self, input_shape=None)`: Generates a list of Keras layers from the block settings. Optionally accepts input shape for the first layer.


### `library_compute_stats.py`

Computes evaluation metrics for classification performance.

#### 🔍 Purpose

To convert predictions to class labels (if needed) and compute core classification metrics.

#### 🔧 Key Function
  - `compute_descriptors(y_true, y_pred)`: Calculates accuracy, precision, recall, and F1-score for the model predictions.


### `library_load_and_split_data.py`

Handles data loading, normalization, and stratified K-Fold preparation for session-based IDX-formatted datasets.

#### 🔍 Purpose
Load raw IDX sessions and labels, flatten 28×28 matrices into 1D vectors, normalize inputs, reshape to Conv1D format (samples, 784, 1), optionally one-hot encode labels, and prepare train/validation splits.

> ⚠️ **Note:** You must provide your own `.idx3` and `.idx1` files. The file paths must be manually updated in the script.

#### 🔧 Key Functions

  - `read_idx3(filename)`: Reads and returns session data from an IDX3 file.
  - `read_idx1(filename)`: Reads and returns label data from an IDX1 file.
  - `load_and_preprocess_data(num_classes, test_size=0.1, encode_labels=False)`: Loads IDX data, and splits into train/validation/test sets.
  - `preprocess_data(X, y, num_classes=11, encode_labels=False)`: Normalizes inputs to [0,1], reshapes to (samples, 784, 1), and one-hot encodes the labels if requested.
  - `fold_index(X, y, num_splits=2)`: Generates stratified K-Fold indices for cross-validation.
  - `get_fold_split(X, Y, folds, i_fold)`: Retrieves training and validation splits for a specific fold.

### 🧩 Dependencies

#### 📦 External Packages (install via `pip`)
- `tensorflow` – Deep learning framework (includes Keras)
- `keras-flops` – Estimates model FLOPs
- `scikit-learn` – Utilities for K-Fold, metrics, etc.
- `numpy` – Numerical computations

#### 📁 Standard Python Libraries (built-in)
- `json`
- `os`
- `copy`
- `struct`
- `gc`
- `random`
- `datetime`
- `time`

#### 📚 Internal Modules (provided in this repository)
- `Library_NAS`
- `Library_Net`
- `Library_Block`
- `Library_load_and_split_data`
- `Library_compute_stats`

