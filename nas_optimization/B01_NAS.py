import Library_NAS
import Library_load_and_split_data
import numpy as np
from tensorflow.keras.utils import to_categorical
from datetime import datetime

is_train = True
is_train_proxy = True
num_classes = 11
test_size = 0.1

# Loading and preprocessing data without encoding labels for stractified folds generation
X_train_val, y_train_val, X_test, y_test = Library_load_and_split_data.load_and_preprocess_data(num_classes, test_size, encode_labels=False)

# Generating stratified K-fold indices
folds = Library_load_and_split_data.fold_index(X_train_val, y_train_val)

# Convert labels to one-hot encoding after creating folds
y_train_val = to_categorical(y_train_val, num_classes)
y_test = to_categorical(y_test, num_classes)


# Pack the preloaded data for NAS
preloaded_data = (X_train_val, y_train_val, X_test, y_test)

# Create a unique timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
nas_saver_name = f"NAS_logger_{timestamp}"
partial_saver_name = f"Partial_saver_logger_{timestamp}"

# Initialize the NAS with specified parameters, including hardware constraints
NAS = Library_NAS.NAS(
    is_train_proxy=is_train_proxy,
    is_train=is_train,
    max_depth_father=5,
    max_depth=8,
    check_hw=True,  # Enabling hardware checks
    params_thr=220000,
    flops_thr=10000000,
    flash_thr=900000,  # 0.9 MB,  # Flash memory threshold in bytes
    ram_thr=100500,  # 100.5 kB
    n_generations=70,
    n_child=7,
    n_mutations=1,
    partial_save_steps=5,
    smart_start=True,
    is_random_walk=True,

    nas_saver_name=nas_saver_name,
    partial_saver_name=partial_saver_name,
    preloaded_data=preloaded_data,  # Providing preloaded data
    use_full_training=True
)

# Run the NAS process using the generated fold indices
generated_net, best_absolute = NAS.run_NAS(folds)
