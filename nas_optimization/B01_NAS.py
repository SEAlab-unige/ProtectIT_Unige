import Library_NAS
import Library_load_and_split_data
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime

is_train = False # Full training mode
is_train_proxy = True # Proxy training mode
num_classes = 11
test_size = 0.1

# === Load and preprocess the data (without label encoding for stratified folds) ===
X_train_val, y_train_val, X_test, y_test = Library_load_and_split_data.load_and_preprocess_data(num_classes, test_size, encode_labels=False)

# Generating stratified K-fold indices
folds = Library_load_and_split_data.fold_index(X_train_val, y_train_val)

# === Convert labels to one-hot encoding (after folds have been created) ===
y_train_val = to_categorical(y_train_val, num_classes)
y_test = to_categorical(y_test, num_classes)


# === Bundle data into one object to pass to the NAS engine ===
preloaded_data = (X_train_val, y_train_val, X_test, y_test)

# === Create unique filenames for logging this NAS run ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nas_saver_name = f"NAS_logger_{timestamp}"
partial_saver_name = f"Partial_saver_logger_{timestamp}"

# === Initialize NAS optimizer with configurable settings ===
NAS = Library_NAS.NAS(
    is_train_proxy=is_train_proxy,       # Use proxy training (faster)
    is_train=is_train,                   # Use full training (slower)

    # === Architecture depth constraints ===
    max_depth_father=5,                 # Max depth of initial parent architecture
    max_depth=5,                        # Max depth allowed for mutated children

    # === Hardware constraints ===
    check_hw=True,                      # Enforce hardware constraint checks
    params_thr=120000,                  # Max number of trainable parameters
    flops_thr=11000000,                 # Max allowed FLOPs (floating point operations)
    flash_thr=480000,                   # Max flash size in bytes (e.g., 480 kB)
    ram_thr=88000,                      # Max RAM usage in bytes (e.g., 88 kB)
    max_tens_thr=22000,                 # Max size of intermediate tensors

    # === NAS evolution strategy ===
    n_generations=100,                  # Number of generations to evolve
    n_child=10,                         # Number of child networks per generation
    n_mutations=2,                      # Number of mutations applied to each child

    partial_save_steps=5,              # Save intermediate results every N generations
    smart_start=True,                  # Initialize with hand-crafted blocks
    is_random_walk=True,               # Apply random mutation behavior

    # === Logging and I/O ===
    nas_saver_name=nas_saver_name,               # Log file for the full NAS run
    partial_saver_name=partial_saver_name,       # Log file for partial saves
    preloaded_data=preloaded_data,               # Pre-split and preprocessed data
    use_full_training=False                      # Choose between full/proxy training during NAS
)

# === Run the full NAS optimization loop ===
generated_net, best_absolute = NAS.run_NAS(folds)
