import random
import copy
import numpy as np
import Library_Net
import Library_Block
import Library_load_and_split_data
import os
from datetime import datetime
import time

class NAS:
    # Initialize NAS with training flags, thresholds, search settings, and logging.
    def __init__(self, is_train_proxy, is_train, max_depth_father=3, max_depth=10, check_hw=True, params_thr=np.inf, flops_thr=np.inf, max_tens_thr=np.inf, flash_thr=np.inf, ram_thr=np.inf,
                 n_generations=50, n_child=5, n_mutations=1, partial_save_steps=5, smart_start=True, is_random_walk=True,
                 nas_saver_name="NAS_logger", partial_saver_name="Partial_saver_logger", preloaded_data=None, use_full_training=False):
        self.use_full_training = use_full_training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nas_saver_name = f"{nas_saver_name}_{timestamp}"
        self.partial_saver_name = f"{partial_saver_name}_{timestamp}"
        self.max_depth_father = max_depth_father
        self.max_depth = max_depth
        self.is_train_proxy = is_train_proxy
        self.is_train = is_train
        self.preloaded_data = preloaded_data

        self.n_generations = n_generations
        self.n_child = n_child
        self.n_mutations = n_mutations
        self.nas_saver_name = nas_saver_name
        self.log_message('-------------------------NEW NAS--------------------', mode='w')

        self.check_hw = check_hw
        self.flops_thr = flops_thr
        self.flash_thr = flash_thr
        self.ram_thr = ram_thr
        self.params_thr = params_thr
        self.max_tens_thr = max_tens_thr

        self.partial_save_steps = partial_save_steps
        self.smart_start = smart_start
        self.is_random_walk = is_random_walk


        self.partial_saver_name = partial_saver_name
        self.base_dropout_rate = 0.1
        self.max_dropout_rate = 0.5

        #self.is_dist = is_dist

    # Run the full NAS loop across generations to evolve networks.
    # Starts from an initial parent and selects the best child each generation.
    def run_NAS(self, folds):
        # Ensure data is loaded from outside this function
        if self.preloaded_data is None:
            print("Error: Data must be preloaded and provided to NAS.")
            return

        absolute_best_score = 0
        # Initialize the blocks based on the provided configurations
        b1 = Library_Block.Block(
            n_filters=27, kernel_size=3, activation="relu", padding="same",
            is_pool=True, pool_size=2, input_size=784, is_dropout=True, dropout_rate=0.1,
            stride=4, nas_saver_name=self.nas_saver_name, is_max_pool=True, is_avg_pool=False
        )
        b1_output_size = b1.calculate_output_size()

        b2 = Library_Block.Block(
            n_filters=114, kernel_size=4, activation="relu", padding="same",
            is_pool=True, pool_size=3, input_size=b1_output_size, is_dropout=True, dropout_rate=0.2,
            stride=4, nas_saver_name=self.nas_saver_name, is_max_pool=False, is_avg_pool=True
        )

        # Define the parent network
        parent = Library_Net.Net([b1, b2], nas_saver_name=self.nas_saver_name,
                                 preloaded_data=self.preloaded_data)
        parent.short_description()

        #else:
            #parent = self.generate_random_network_and_control()

        for i_gen in range(self.n_generations):
            self.log_message('GENERATION ' + str(i_gen) + '\n')

            parent, gen_best_score, test_metrics = self.one_nas_step(parent, folds)

            # Delay after processing each generation
            print(f"Delaying 60 seconds for cooling down after generation {i_gen}")
            time.sleep(60)  # Delay for 200 seconds
            if (gen_best_score > absolute_best_score):
                absolute_best_score = gen_best_score
                best_network = copy.deepcopy(parent)
                self.log_message('-----------------------------------\n')
                self.log_message('NEW ABSOLUTE BEST FOUND AT GENERATION ' + str(i_gen) + '\n')

                self.save_partial(parent, best_network, i_gen, gen_best_score, absolute_best_score)
                #self.log_message(f'Test Metrics: {test_metrics}\n')  # Log test metrics for informational purposes
                best_network.dump()
                best_network.short_description()
        return parent, best_network

    # Train all children in a generation using full or proxy routine.
    # Returns validation scores and test metrics for each child
    def train_generation(self, child_set, folds):
        gen_per = []
        average_test_metrics_per_child = []  # To store average test metrics for each child

        for net in child_set:
            if self.use_full_training:
                # Use the full training routine, which returns a list of (average validation score, test_metrics)
                train_results = net.train_routine(self.is_train, folds)
                # Collect the average validation scores from the returned results
                avg_val_score = sum(result[0] for result in train_results) / len(train_results)
                # Calculate average test metrics across all folds
                total_test_metrics = np.zeros(5)  # Assuming test_metrics are in the form [acc, prec, rec, f1]
                for result in train_results:
                    total_test_metrics += np.array(result[1])  # Assuming result[1] is a list/tuple of metrics
                avg_test_metrics = total_test_metrics / len(train_results)

                gen_per.append(avg_val_score)
                average_test_metrics_per_child.append(avg_test_metrics.tolist())  # Convert to list for consistency
            else:
                # Since proxy_train_routine now returns only a single tuple
                proxy_result = net.proxy_train_routine(is_train_proxy=self.is_train_proxy, selected_fold_index=0, validation_split=0.11, folds=folds)[0]
                avg_val_score, avg_test_acc = proxy_result

                gen_per.append(avg_val_score)
                average_test_metrics_per_child.append([avg_test_acc])  # Store it as a list of one element

            # Delay after training each child network
            print(f"Delaying 10 seconds for cooling down after training child network")
            time.sleep(10)  # Delay for 60 seconds
        return gen_per, average_test_metrics_per_child

    # Select the best-performing child network based on validation accuracy.
    # Logs test metrics and returns the best child and its score.
    def best_child_selection(self, child_set, gen_per, test_metrics_list):
        self.log_message('Selection Starts: \n')

        best = -1  # Initialize with -1, assuming accuracy ranges [0, 1]
        best_child = None
        best_test_metrics = None  # To store the test metrics of the best child

        for i_c, score in enumerate(gen_per):
            self.log_message(f"Child {i_c} Val Acc: {score}\n")
            if score > best:  # Looking for higher validation accuracy
                best = score
                best_child = copy.deepcopy(child_set[i_c])
                best_test_metrics = test_metrics_list[i_c]  # Retrieve corresponding test metrics for the best child

        self.log_message(f"Selection ended. Best Val Acc: {best}\n")
        if len(best_test_metrics) == 5:
            self.log_message(
                f"Best Child Test Metrics: Accuracy={best_test_metrics[0]}, Precision={best_test_metrics[1]}, Recall={best_test_metrics[2]}, F1={best_test_metrics[3]}, Average={best_test_metrics[4]}\n")
        else:
            self.log_message(f"Best Child Test Metrics: Test Accuracy={best_test_metrics[0]}\n")

        if best_child is not None:
            best_child.dump()  # Log the architecture of the best child
            best_child.short_description()
        return best_child, best

    # Perform one NAS generation step: mutation, training, and selection.
    def one_nas_step(self, parent, folds):
        child_set = self.new_generation(parent, self.preloaded_data)
        gen_per, test_metrics_list = self.train_generation(child_set, folds)
        best_child, best_score = self.best_child_selection(child_set, gen_per, test_metrics_list)
        return best_child, best_score, test_metrics_list

    # Generate a new population of child networks by mutating the parent.
    # Applies multiple mutations per child if specified.
    def new_generation(self, parent, preloaded_data):
        child_set =[]
        if(self.is_random_walk == False):
            child_set.append(copy.deepcopy(parent))
            self.log_message(" Parent; \n")
            parent.dump()
            self.log_message("\n")

        for i_child in range(self.n_child):
            self.log_message(f" CHILD i-th = {i_child}; \n")
            child = self.mutate_network_and_control(parent, preloaded_data if preloaded_data else self.preloaded_data)
            if (self.n_mutations > 1):
                for i_mut in range(self.n_mutations - 1):
                    child = self.mutate_network_and_control(child, preloaded_data if preloaded_data else self.preloaded_data)
            child_set.append(child)
            child.dump()
            self.log_message("\n")
        return child_set

    # Mutate a network with hardware constraint checks if enabled.
    # Only return the child if it passes all hardware thresholds.
    def mutate_network_and_control(self, parent, preloaded_data):
        if self.check_hw:
            check_pass = False
            while not check_pass:
                child_net = self.mutate_network(parent, preloaded_data)

                # Validate and adjust child blocks
                for block in child_net.block_list:
                    # Strict constraint: Avoid "valid" padding for small input sizes
                    if block.input_size < 32:
                        block.padding = "same"

                    # Switch to "same" padding if "valid" is infeasible
                    if block.padding == "valid" and block.input_size < block.kernel_size:
                        block.padding = "same"

                    # Enforce kernel size and stride constraints
                    block.kernel_size = min(block.kernel_size, block.input_size)
                    block.stride = min(block.stride, block.input_size)

                # Hardware checks
                hw_meas = child_net.hw_measures()
                flash_size = hw_meas[3]
                ram_size = hw_meas[4]
                flops = hw_meas[2]
                n_params = hw_meas[0]
                max_tens = hw_meas[1]
                if (
                        flops < self.flops_thr and
                        flash_size < self.flash_thr and
                        ram_size < self.ram_thr and
                        n_params < self.params_thr and
                        max_tens < self.max_tens_thr
                ):
                    check_pass = True
        else:
            child_net = self.mutate_network(parent, preloaded_data)

        return child_net

    # Mutate a network by randomly adding, removing, or changing a block.
    # Applies basic shape correction after mutation.
    def mutate_network(self, parent, preloaded_data):
        self.log_message('MUTATION ')

        parent_blocks = copy.deepcopy(parent.block_list)

        # Perform mutation action: remove, add, or change a block
        act = random.randint(0, 2)  # 0 remove, #1 change, 2 add
        if act == 0:
            self.remove_block(parent_blocks)
        elif act == 1:
            self.add_block(parent_blocks)
        elif act == 2:
            self.change_block(parent_blocks)

            # Ensure the first block has the correct input_size
        if parent_blocks:
            parent_blocks[0].input_size = 784  # Reset to original dataset input size

        child_block = self.correct_blocklist(parent_blocks)
        self.log_message(" net_depth = " + str(len(child_block)) + "; ")

        # Pass the preloaded data to the newly created Net instance
        child_net = Library_Net.Net(child_block, nas_saver_name=self.nas_saver_name, preloaded_data=preloaded_data)

        child_net.trained_fully = parent.trained_fully

        return child_net

    # Generate a random block with constraints based on input size and position.
    def generate_random_block(self, input_size, block_index, current_architecture_depth):
        # Define parameter ranges or constraints
        n_filters_range = (1, 140)
        kernel_size = random.randint(1, min(7, input_size))  # Ensure kernel size <= input size
        stride_range = (1, min(input_size, 6))  # Stride must not exceed input_size

        base_dropout_rate = 0.1
        max_dropout_rate = 0.5

        # Dropout rate increases with depth
        if current_architecture_depth > 1:
            dropout_rate = base_dropout_rate + (max_dropout_rate - base_dropout_rate) * block_index / (
                        current_architecture_depth - 1)
        else:
            dropout_rate = max_dropout_rate

        dropout_rate = round(dropout_rate, 2)

        # Strict constraint: Avoid "valid" padding for small input sizes
        if input_size < 32:
            padding = "same"
        else:
            padding = "same" if random.getrandbits(1) else "valid"

        # Switch to "same" padding if "valid" is infeasible
        if padding == "valid" and kernel_size > input_size:
            padding = "same"

        # Validate kernel size
        kernel_size = min(kernel_size, input_size)

        # Prevent pooling in the first block
        if block_index == 0:
            is_pool = False
            pool_size = None
            is_max_pool = False
            is_avg_pool = False
        else:
            is_pool = bool(random.getrandbits(1))

            # Calculate pool size range dynamically
            min_pool_size = 2
            max_pool_size = min(3, input_size)

            # Ensure pool_size_range is valid
            if max_pool_size < min_pool_size:
                is_pool = False
                pool_size = None
                is_max_pool = False
                is_avg_pool = False
            else:
                pool_size = random.randint(min_pool_size, max_pool_size) if is_pool else None
                is_max_pool = bool(random.getrandbits(1)) if is_pool else False
                is_avg_pool = not is_max_pool if is_pool else False

        # Stride
        stride = random.randint(*stride_range)

        return Library_Block.Block(
            n_filters=random.randint(*n_filters_range),
            kernel_size=kernel_size,
            activation="relu",
            padding=padding,  # Randomly chosen padding
            is_pool=is_pool,
            pool_size=pool_size,
            is_max_pool=is_max_pool,
            is_avg_pool=is_avg_pool,
            input_size=input_size,
            is_dropout=True,
            dropout_rate=dropout_rate,
            stride=stride,
            nas_saver_name=self.nas_saver_name
        )

    # Remove a random non-input block from the architecture.
    # Fallback: mutate if removal not allowed.
    def remove_block(self, parent_blocks):
        self.log_message(' REMOVE')
        if len(parent_blocks) > 1:
            i_del = random.randint(1, len(parent_blocks) - 1)  # Start from 1 to preserve input layer
            parent_blocks.pop(i_del)
            self.recalculate_dropout_rates(parent_blocks)# Update dropout rates
            self.correct_blocklist(parent_blocks)
        else:
            self.log_message(" FORBIDDEN -> CHANGED")
            i_mod = random.randint(0, len(parent_blocks) - 1)
            input_size = parent_blocks[i_mod].input_size if parent_blocks else 784
            parent_blocks[i_mod] = self.generate_random_block(input_size, i_mod, len(parent_blocks))
            self.recalculate_dropout_rates(parent_blocks)  # Update dropout rates
            self.correct_blocklist(parent_blocks)
        return parent_blocks

    # Replace a random block in the architecture with a new one.
    def change_block(self, parent_blocks):
        self.log_message(' CHANGE')
        if parent_blocks:  # Ensure there's at least one block to modify
            i_mod = random.randint(0, len(parent_blocks) - 1)
            input_size = parent_blocks[i_mod].input_size
            parent_blocks[i_mod] = self.generate_random_block(input_size, i_mod, len(parent_blocks))
            self.recalculate_dropout_rates(parent_blocks)
            self.correct_blocklist(parent_blocks)
        return parent_blocks

    # Add a new block at a random position, or mutate if max depth is reached.
    def add_block(self, parent_blocks):
        self.log_message(' ADD')
        if len(parent_blocks) < self.max_depth:
            i_add = random.randint(0, len(parent_blocks))
            input_size = parent_blocks[-1].output_size if parent_blocks else 784
            new_block = self.generate_random_block(input_size, len(parent_blocks), len(parent_blocks) + 1)
            if i_add == len(parent_blocks):
                parent_blocks.append(new_block)
            else:
                parent_blocks.insert(i_add, new_block)
            self.recalculate_dropout_rates(parent_blocks)  # Update dropout rates
            self.correct_blocklist(parent_blocks)
        else:
            self.log_message(' ADD FORBIDDEN, CHANGED')
            i_mod = random.randint(0, len(parent_blocks) - 1)
            input_size = parent_blocks[i_mod].output_size if parent_blocks else 784
            parent_blocks[i_mod] = self.generate_random_block(input_size, i_mod, len(parent_blocks))
            self.recalculate_dropout_rates(parent_blocks)  # Update dropout rates
            self.correct_blocklist(parent_blocks)

        return parent_blocks

    # Update dropout rates for all blocks based on architecture depth.
    def recalculate_dropout_rates(self, blocks):
        current_architecture_depth = len(blocks)
        for i, block in enumerate(blocks):
            if current_architecture_depth > 1:
                dropout_rate = self.base_dropout_rate + (self.max_dropout_rate - self.base_dropout_rate) * i / (
                        current_architecture_depth - 1)
            else:
                dropout_rate = self.max_dropout_rate
            block.dropout_rate = round(dropout_rate, 2)

    # Apply constraints and correct shape/behavior across all blocks.
    # Ensures valid padding, stride, and pooling logic.
    def correct_blocklist(self, child_blocks):
        forbid_pool = False

        for i in range(len(child_blocks)):
            # For the first block, explicitly disable pooling
            if i == 0:
                child_blocks[i].is_pool = False
                child_blocks[i].is_max_pool = False
                child_blocks[i].is_avg_pool = False

            # Set input_size dynamically for subsequent blocks
            if i > 0:
                child_blocks[i].input_size = child_blocks[i - 1].output_size

            # Strict constraint: Avoid "valid" padding if input size is small
            if child_blocks[i].input_size < 32:
                child_blocks[i].padding = "same"

            # Switch to "same" padding if "valid" is infeasible
            if child_blocks[i].padding == "valid" and child_blocks[i].input_size < child_blocks[i].kernel_size:
                child_blocks[i].padding = "same"

            # Enforce kernel size constraints
            child_blocks[i].kernel_size = min(child_blocks[i].kernel_size, child_blocks[i].input_size)

            # Enforce stride constraints
            child_blocks[i].stride = min(child_blocks[i].stride, child_blocks[i].input_size)

            # Calculate output_size using the block's method for accurate dimensionality
            child_blocks[i].output_size = child_blocks[i].calculate_output_size()

            # Enforce the forbid_pool rule if input_size < 32
            if child_blocks[i].input_size < 32:
                forbid_pool = True

            if forbid_pool:
                child_blocks[i].is_pool = False
                child_blocks[i].is_max_pool = False
                child_blocks[i].is_avg_pool = False
                # Recalculate output_size as pooling is disabled
                child_blocks[i].output_size = child_blocks[i].calculate_output_size()
            elif child_blocks[i].is_pool:
                # Ensure pooling is valid
                if child_blocks[i].output_size < child_blocks[i].pool_size:
                    child_blocks[i].is_pool = False
                    child_blocks[i].is_max_pool = False
                    child_blocks[i].is_avg_pool = False
                    # Recalculate output_size as pooling is disabled
                    child_blocks[i].output_size = child_blocks[i].calculate_output_size()

        return child_blocks

    # Save parent and best model of the current generation to disk.
    # Logs save status and generation info.
    import os
    def save_partial(self, parent, best_network, i_gen, gen_best_score, absolute_best_score):
        if not os.path.exists("../sessions_nas/Models"):
            os.makedirs("../sessions_nas/Models")

        model_parent = parent.ins_keras_model()
        model_parent.save(f"./Models/{self.nas_saver_name}_Partial_parent")

        model_best_network = best_network.ins_keras_model()
        model_best_network.save(f"./Models/{self.nas_saver_name}_Partial_best_network")

        # Log information about the saving process
        self.log_message(f"Saved parent model at generation {i_gen} with score {gen_best_score:.2f}")
        self.log_message(f"Saved best network at generation {i_gen} with absolute best score {absolute_best_score:.2f}")

    # Log a message to the NAS experiment log file.
    # Mode 'a' = append, 'w' = overwrite.
    def log_message(self, message, mode='a'):
        with open(self.nas_saver_name + '.txt', mode) as log_file:
            log_file.write(message + '\n')
