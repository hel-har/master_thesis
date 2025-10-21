# %%
## Python libraries import

from device_selection import get_pos_neg_apriori 
from simclr_models import AdaptationLayer
import os
import random
import pickle
import scipy
import datetime
import time
import argparse
import numpy as np
import copy
import tensorflow as tf
from common_parser import get_parser
# %%

import sys
import simclr_models
import simclr_utitlities
import transformations
import statistics as stats
import visual_utils
from PIL import Image

# %%
## Data loading script import
import load_data

## Loss function import
from loss_fn import *

tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

def get_random_shuffle_indices(length, seed):
    rng = np.random.default_rng(seed=seed)
    index_list = np.arange(length, dtype=int)
    rng.shuffle(index_list)
    return index_list

def shuffle_array(array, seed, inplace=True):
    if inplace:
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(array)
    else:
        length = array.shape[0]
        rng = np.random.default_rng(seed=seed)
        index_list = np.arange(length, dtype=int)
        rng.shuffle(index_list)
        return array[index_list]

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_group_held_out_users(all_users, group_index, num_groups):
    group_length = round(len(all_users) / num_groups)
    groups = [all_users[i * group_length: (i + 1) * group_length] if i < num_groups - 1 else all_users[i * group_length:] for i in range(num_groups)]
    return groups[group_index]

def get_adaptation_group(groups_list, device):
    for i, group in enumerate(groups_list):
        for j, d in enumerate(group):
            if d==device: 
                return (i, j)
    return ()

class BatchedRandomisedDataset:
    def __init__(self, data, batch_size, seed=42, randomised=True, axis=0, post_process_func=None, name=""):
        self.name = name
        self.data = data
        self.batch_size = batch_size
        self.axis = axis
        self.seed = seed
        self.data_len = data.shape[self.axis]
        self.num_batches = ceiling_division(self.data_len, batch_size)
        self.randomised = randomised
        self.rng = np.random.default_rng(seed=seed)
        self.post_process_func = post_process_func
        # if post_process_func is None:
        #     post_process_func = lambda x: x
        
    def reset_dataset(self):
        if self.randomised:
            index_list = np.arange(self.data_len, dtype=int)
            self.rng.shuffle(index_list)
            self.shuffled_dataset = self.data[index_list]
            self.output_dataset = self.shuffled_dataset
        else:
            self.output_dataset = self.data
        if not self.axis == 0:
            self.output_dataset = np.moveaxis(self.output_dataset, self.axis, 0)
        self.i = 0

    def __len__(self):
        return self.num_batches
    
    # def __iter__(self):
    #     self.reset_dataset()
    #     return self

    # def __next__(self):
    #     if self.i < self.num_batches:
    #         i = self.i
    #         self.i += 1
    #         if self.axis == 0:
    #             return self.post_process_func(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size])
    #         else:
    #             return self.post_process_func(np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis + 1))
    #     else:
    #         raise StopIteration

    def __iter__(self):
        self.reset_dataset()

        def gen():
            if self.post_process_func is None:
                if self.axis == 0:
                    for i in range(self.num_batches):
                        yield self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size]
                else:
                    for i in range(self.num_batches):
                        yield np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis)
            else:
                if self.axis == 0:
                    for i in range(self.num_batches):
                        yield self.post_process_func(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size])
                else:
                    for i in range(self.num_batches):
                        yield self.post_process_func(np.moveaxis(self.output_dataset[i * self.batch_size : (i + 1) * self.batch_size], 0, self.axis))
        return gen()

class SequenceEagerZippedDataset(tf.keras.utils.Sequence):
    def __init__(self, batched_randomised_datasets, stack_batches=True, stack_axis=0):
        self.datasets = batched_randomised_datasets
        self.stack_batches = stack_batches
        self.stack_axis = stack_axis
        self.reset_dataset()

    def reset_dataset(self):
        if self.stack_batches:
            self.data = [
                np.stack(zipped_batch, axis=self.stack_axis) for zipped_batch in zip(*tuple(self.datasets))
            ]
        else:
            self.data = [
                zipped_batch for zipped_batch in zip(*tuple(self.datasets))
            ]

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def on_epoch_end(self):
        self.reset_dataset()

class ZippedDataset:
    def __init__(self, batched_randomised_datasets, stack_batches=True, stack_axis=0):
        self.datasets = batched_randomised_datasets
        self.stack_batches = stack_batches
        self.stack_axis = stack_axis

    def __len__(self):
        return min([len(dataset) for dataset in self.datasets])
    
    # def reset_dataset(self):
    #     self.iters = [iter(dataset) for dataset in self.datasets]
    # def __iter__(self):
    #     self.reset_dataset()
    #     return self

    # def __next__(self):
    #     batch = [next(it) for it in self.iters]
    #     if self.stack_batches:
    #         return np.stack(batch, axis=self.stack_axis)
    #     else:
    #         return tuple(batch)

    def __iter__(self):
        def gen():
            if self.stack_batches:
                for zipped_batch in zip(*tuple(self.datasets)):
                    yield np.stack(zipped_batch, axis=self.stack_axis)
            else:
                for zipped_batch in zip(*tuple(self.datasets)):
                    yield zipped_batch
        return gen()

class ConcatenatedDataset:
    def __init__(self, batched_randomised_datasets):
        self.datasets = batched_randomised_datasets
        self.num_datasets = len(self.datasets)

    # def reset_dataset(self):
    #     self.iters = [iter(dataset) for dataset in self.datasets]
    #     self.iter_i = 0
    
    # def __iter__(self):
    #     self.reset_dataset()
    #     return self

    # def __next__(self):
    #     batch = None
    #     while (batch is None):
    #         try:
    #             batch = next(self.iters[self.iter_i])
    #         except StopIteration:
    #             batch = None
    #             self.iter_i += 1
    #             if self.iter_i == self.num_datasets:
    #                 raise StopIteration
    #     return batch
    
    def __iter__(self):
        def gen():
            for dataset in self.datasets:
                for batch in dataset:
                    yield batch
        return gen()

train_devices, positive_indices, negative_indices = [], [], []
def train(dataset_full, args):

    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)
    
    input_shape = dataset_full.input_shape

    # tf_train_full = tf_dataset_full.ds_train[args.train_device].map(lambda x, y, i: (x, y))

    # output_shape = len(np.unique([y for x, y in tf_train_full])) # Infer number of classes from training data (slow)
    output_shape = len(dataset_full.info['session_list'])
    #print("input shape", input_shape)
    #print("output shape", output_shape)


    # %%
    ## Setup working folder

    # working_directory = args.working_directory if args.working_directory.endswith("/") else args.working_directory + "/"
    # if not os.path.exists(working_directory):
    #     os.mkdir(working_directory)
    # start_time = datetime.datetime.now()
    # start_time_str = start_time.strftime("%Y%m%d-%H%M%S")

    if args.multi_anchor:
        working_directory = os.path.join(args.working_directory, 'multi_anchor', args.multi_anchor_training_mode, args.exp_name, args.training_mode)
    elif args.multi_anchor_batches:
        working_directory = os.path.join(args.working_directory, 'multi_anchor_batches', args.multi_anchor_training_mode, args.exp_name, args.training_mode)
    else:
        working_directory = os.path.join(args.working_directory, args.train_device, args.exp_name, args.training_mode)
    
    if args.adaptation_layer:
        working_directory = os.path.join(working_directory, args.adaptation_architecture)
    
    working_directory = os.path.join(working_directory, args.eval_mode)



    if args.generalizability:
        working_directory = os.path.join(working_directory, 'generalizability')
    else: 
        working_directory = os.path.join(working_directory, 'adaptability')

    if not os.path.exists(working_directory):
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)


    if not hasattr(args, 'start_time'):
        args.start_time = str(int(datetime.datetime.now().timestamp()))
    if not hasattr(args, 'run_name'):
        if args.multi_anchor:
            args.run_name = f"run-{run_id}-{args.training_mode}-{args.multi_anchor_list}-{args.multi_anchor_training_mode}"
        elif args.multi_anchor_batches:
            args.run_name = f"run-{run_id}-{args.training_mode}-multi_anchor_batches-{args.multi_anchor_training_mode}"
        else:
            args.run_name = f"run-{run_id}-{args.training_mode}-{args.train_device}"    #f"run-{args.start_time}"
        if args.generalizability:
            args.run_name += f"-{args.eval_device}-generalizability"

    ## Model Architecture

    if args.trained_model_path is None:
        if args.model_arch == '1d_conv':
            if args.adaptation_layer:
                assert args.multi_anchor or args.multi_anchor_batches, "args.adaptation layer can only be run with multi-anchor training, use --multi_anchor or --multi_anchor_batches"
                num_adaptation_layers = len(args.adaptation_groups)
            else:
                num_adaptation_layers = 0
            base_model, last_freeze_layer = simclr_models.create_base_model(input_shape, num_adaptation_layers, adaptation_architecture=args.adaptation_architecture, model_name="1d_conv")
            args.trained_model_last_freeze_layer = last_freeze_layer
            print(f"Model layers: {len(base_model.layers)}, Last freeze layer: {args.trained_model_last_freeze_layer}")
            print(f"Last layer to be frozen: {base_model.layers[args.trained_model_last_freeze_layer]}")
    else:
        base_model = tf.keras.models.load_model(args.trained_model_path, custom_objects={"AdaptationLayer": AdaptationLayer})
        last_freeze_layer = args.trained_model_last_freeze_layer

    # %%
    ## Training hyperparameters

    if args.training_mode != 'none':
        batch_size = args.training_batch_size
        decay_steps = args.training_decay_steps
        epochs = args.training_epochs
        temperature = args.contrastive_temperature
        initial_lr = args.learning_rate

    # %%

    # %%
        ## Prepare for training (creation of learning rate decay, optimizer, neural network)

        tf.keras.backend.set_floatx('float32')

        if args.learning_rate_decay == 'cosine':
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)
        elif args.learning_rate_decay == 'none':
            lr_decayed_fn = initial_lr

        if args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)
        elif args.optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr_decayed_fn)
        elif args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)


        if args.data_aug == 'none':
            transformation_function = lambda x: x
        elif args.data_aug == 'rotate':
            transform_func_structs = [
                ([0,1,2], transformations.rotation_transform_vectorized),
                ([3,4,5], transformations.rotation_transform_vectorized)
            ]
            transformation_function = simclr_utitlities.generate_slicing_transform_function(
                transform_func_structs, slicing_axis=2, concatenate_axis=2
            )
        elif args.data_aug == 'sensor_noise':
            transformation_function = lambda x: transformations.scaling_transform_vectorized(transformations.noise_transform_vectorized(x))
        

    # %%
        if args.training_mode == 'multi':

            simclr_model = base_model
            simclr_model.summary()
            ## Multi-device Training
            print('='*20+"Multi-device Training"+'='*20)

            # Design notes:
            ## The training loop is expected to function as follows:
            ## It samples n samples from each device (n = batch size)
            ## Each of these n samples are passed through the model
            ## And then the loss function is called, which should accept the anchor embeddings (n x e), positive embeddings (p x n x e) and negative embeddings (q x n x e)
            ## p = number of positive devices, q = number of negative devices
            ## The model accepts a (n x t x c) windowed time series, and outputs embeddings in shape (n x e)
            ## To allow for better generalizability, the sampling should be agnostic to the positive/negative assignments (all devices are sampled at the same time)
            ## And when obtaining the loss, it should allow for different assignments by concatenating different device embeddings based on arguments passed as positive/negative indices

            # Index mappings
            # all_devices = ['forearm', 'thigh', 'head', 'chest', 'upperarm', 'waist', 'shin']
            global train_devices, positive_indices, negative_indices
            train_devices = [args.train_device]
            train_devices.extend(args.positive_devices)
            train_devices.extend(args.negative_devices)

            all_devices = list(dataset_full.info['device_list'])
        
            if (len(args.positive_devices)==0 or len(args.negative_devices)==0) or args.generalizability: # device_selection will be called
                train_devices = list(dataset_full.info['device_list'])
                #all_devices.remove(args.train_device)
                #all_devices = [args.train_device] + all_devices
                if args.generalizability: # generalizability evaluation is active, hence the evaluation device is removed from the training
                    if(args.eval_device in train_devices):
                        train_devices.remove(args.eval_device)
                        print(f"Generalizabiility enabled, leaving eval_device ({args.eval_device}) out while training.")
                if args.multi_anchor or args.multi_anchor_batches:
                    print("Anchors:", args.multi_anchor_list, "Evaluation:", args.multi_eval_list, "Training Devices:", train_devices)
                else:
                    print("Anchor:", args.train_device, "Evaluation:", args.eval_device, "Training Devices:", train_devices)

                
            anchor_index_list = dict()
            anchor_index_list['train_devices'] = [train_devices.index(args.train_device)]
            anchor_index_list['adapt_groups'] = [get_adaptation_group(args.adaptation_groups,args.train_device)[0]]
            if args.multi_anchor: #args.multi_anchor_batches:
                for d in args.multi_anchor_list:
                    if d != args.train_device:
                        anchor_index_list['train_devices'].append(train_devices.index(d))
                        anchor_index_list['adapt_groups'].append(get_adaptation_group(args.adaptation_groups,d)[0])
            positive_indices = np.arange(len(args.positive_devices)) + 1
            negative_indices = np.arange(len(args.negative_devices)) + 1 + len(args.positive_devices)

            print(f"Anchor indices: {anchor_index_list}")

            training_set_device = [(np.concatenate([dataset_full.device_user_ds[d][u][0] for u in dataset_full.device_user_ds[d] if u not in held_out_users], axis=0)) for d in train_devices]
            # training_full_size = len(training_set_device[0])
            # train_samples_count = int(training_full_size * args.take)

            if args.dynamic_device_selection==0 and (len(args.positive_devices)==0 or len(args.negative_devices)==0):
                training_set_stacked = np.stack(training_set_device, axis=0)
                tf_train_contrast = BatchedRandomisedDataset(training_set_stacked, batch_size, randomised=False, axis=1, name="distances")
                positive_indices, negative_indices, distances = get_pos_neg_apriori(tf_train_contrast, train_devices, strategy=args.device_selection_strategy)
                print("Computing Pos./Neg. devices apriori.")
                print("Positives:", train_devices[positive_indices], "Negatives:", train_devices[negative_indices])
            else:
                distances = None
                print("Computing Pos./Neg. devices dynamically.")

            

            if args.multi_sampling_mode == 'sync_all':
                # if distances is not None:
                #     distances = tf.convert_to_tensor(distances, dtype=tf.float64)
                pass
            else:
                # Unsynchronised sampling for different groupings of devices     
                if distances is not None:
                    overlap_indices = list(set.intersection(set(positive_indices), set(negative_indices)))
                    overlap_devices = [train_devices[i] for i in overlap_indices] 
                    all_device_length = len(train_devices)

                    #remove the overlap indices from their original order, to prevent them from becoming positive_devices/aligned
                    negative_indices = [x for x in negative_indices if x not in set(overlap_indices)]
                    negative_devices = [train_devices[i] for i in negative_indices]

                    #Allow for duplicate anchor dataset for negative sampling
                    #Add anchor as a negative device at the end  
                    negative_indices = negative_indices + [*range(all_device_length, all_device_length + len(overlap_indices) +  1, 1)]                    
                    train_devices = train_devices + [args.train_device] + overlap_devices

                    distances[all_device_length] = (min(distances.values())/2.0) 
                    for i,d in enumerate(overlap_indices):
                        distances[all_device_length+i+1] = (distances[d]) #add the distance of the  overlapped element at the end of the list
                        # del distances[d-1] #remove the overlapped element from its original place
                    
                    # distances = tf.convert_to_tensor(distances, dtype=tf.float64)
                            
            user_device_dataset = []
            for u in dataset_full.info['user_list']:
                dataset_per_user = []
                if u not in held_out_users:
                # if args.held_out is None or u != dataset_full.info['user_list'][args.held_out]:
                    for d in train_devices:
                        X = transformation_function(dataset_full.device_user_ds[d][u][0])
                        len_x = X.shape[0]
                        X_shuffled = shuffle_array(X, seed=42, inplace=False) 
                        dataset_per_user.append(X_shuffled[: int(len_x * args.take)])
                    user_device_dataset.append(dataset_per_user)

            tf_train_contrast_list = []
            for user_dataset in user_device_dataset:
                device_dataset_shuffled = []
                for device_index, device_dataset in enumerate(user_dataset):
                                            
                    if args.multi_sampling_mode == 'sync_all':
                        seed = 42
                    elif args.multi_sampling_mode == 'unsync_neg':
                        if device_index == anchor_index_list['train_devices'][0]:
                            seed = 42
                        elif device_index in positive_indices:
                            seed = 42
                        else: 
                            seed = 43 + device_index
                            
                    elif args.multi_sampling_mode == 'unsync_all':
                        seed = 42 + device_index

                    #For Dynamic Device Selection, we want the datasets to be synced here. They will be shuffled later
                    if args.dynamic_device_selection==1 and args.multi_sampling_mode != 'unsync_all':  
                        seed = 42

                    shuffled = BatchedRandomisedDataset(device_dataset, batch_size=batch_size, seed=seed)
                    device_dataset_shuffled.append(shuffled)

                tf_train_contrast_list.append(ZippedDataset(device_dataset_shuffled, stack_batches=True))

            tf_train_concat = ConcatenatedDataset(tf_train_contrast_list)

            if args.adaptation_layer and args.reg_loss:
                print("Regulation loss enabled")
                weighted_loss_function = lambda a_e, p_e, p_w, n_e, n_w, a_data, a_id, adapt_layers: weighted_group_contrastive_loss_with_temp_and_reg(a_e, p_e, p_w, n_e, n_w, a_data, a_id, adapt_layers, temperature=temperature, lambda_reg=1e-4)
            else:
                weighted_loss_function = lambda a_e, p_e, p_w, n_e, n_w: weighted_group_contrastive_loss_with_temp(a_e, p_e, p_w, n_e, n_w, temperature=temperature)



            # if args.dynamic_device_selection==0 and (len(args.positive_devices)==0 or len(args.negative_devices)==0):
            #     positive_indices, negative_indices = get_pos_neg_apriori(tf_train_contrast, all_devices)

            index_mappings = (anchor_index_list, positive_indices, negative_indices)
            if not os.path.exists(f"{working_directory}/models/{args.run_name}.keras"):
                trained_model_save_path = f"{working_directory}/models/{args.run_name}.keras"
                trained_model_low_loss_save_path = f"{working_directory}/models/{args.run_name}_lowest.keras"
                trained_model, epoch_losses = simclr_utitlities.group_supervised_contrastive_train_model(simclr_model, tf_train_concat, transformation_function, optimizer, index_mappings, distances, weighted_loss_function, args.device_selection_strategy, args, weighted=args.weighted_collossl, epochs=epochs, verbose=1, training=True, temporary_save_model_path=trained_model_low_loss_save_path)
                trained_model.save(trained_model_save_path)
                trained_model_save_path = trained_model_low_loss_save_path
            elif args.multi_anchor:
                trained_model_save_path = args.trained_model_path
                trained_model_low_loss_save_path = trained_model_save_path
                trained_model, epoch_losses = simclr_utitlities.group_supervised_contrastive_train_model(simclr_model, tf_train_concat, transformation_function, optimizer, index_mappings, distances, weighted_loss_function, args.device_selection_strategy, args, weighted=args.weighted_collossl, epochs=epochs, verbose=1, training=True, temporary_save_model_path=trained_model_low_loss_save_path)
                trained_model.save(trained_model_save_path)
                trained_model_save_path = trained_model_low_loss_save_path
            else:
                trained_model_save_path = f"{working_directory}/models/{args.run_name}.keras"

        elif args.training_mode == 'supervised':
            
            supervised_model = simclr_models.create_full_classification_model_from_base_model(base_model, output_shape, optimizer=optimizer, model_name="TPN", intermediate_layer=-1, last_freeze_layer=-1)
            
            full_model_save_path = f"{working_directory}/models/{args.run_name}_full.keras"
            best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_model_save_path,
                monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
            )


            if args.baseline=='supervised_all_devices':
                train_devices = list(dataset_full.info['device_list'])
                if args.generalizability:
                    train_devices.remove(args.eval_device)

                training_set_device_X = np.concatenate([dataset_full.device_user_ds[d][u][0] for d in train_devices for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
                training_set_device_y = np.concatenate([dataset_full.device_user_ds[d][u][1] for d in train_devices for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
            else:
                training_set_device_X = np.concatenate([dataset_full.device_user_ds[args.train_device][u][0] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)
                training_set_device_y = np.concatenate([dataset_full.device_user_ds[args.train_device][u][1] for u in sorted(dataset_full.device_user_ds[args.train_device].keys()) if u not in held_out_users], axis=0)

            training_full_size = training_set_device_y.shape[0]
            train_samples_count = int(training_full_size * args.take)
            shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
            training_set_device_X = training_set_device_X[shuffle_indices][:train_samples_count]
            training_set_device_y = tf.keras.utils.to_categorical(training_set_device_y[shuffle_indices][:train_samples_count], num_classes=output_shape)

            train_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[int(train_samples_count * 0.2):], batch_size, post_process_func=transformation_function, seed=42), BatchedRandomisedDataset(training_set_device_y[int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)
            val_split = SequenceEagerZippedDataset([BatchedRandomisedDataset(training_set_device_X[:int(train_samples_count * 0.2)], batch_size, seed=42), BatchedRandomisedDataset(training_set_device_y[:int(train_samples_count * 0.2):], batch_size, seed=42)], stack_batches=False)
            
            callbacks = [best_model_callback]

            
            supervised_model.fit(
                x = train_split,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=val_split
                # verbose=0
            )

            print(full_model_save_path)
            best_supervised_model = tf.keras.models.load_model(full_model_save_path, custom_objects={"AdaptationLayer": AdaptationLayer})

            feature_extractor_model = simclr_models.extract_intermediate_model_from_base_model(best_supervised_model, intermediate_layer=7)
            trained_model_save_path = f"{working_directory}/models/{args.run_name}.keras"
            feature_extractor_model.save(trained_model_save_path)

            if args.eval_mode == 'base_model' or args.eval_mode == 'multi_ft':
                trained_model_save_path = trained_model_save_path
            elif args.eval_mode == 'full_model':
                trained_model_save_path = full_model_save_path

    return trained_model_save_path

def output_tsne_image(working_directory, run_name, eval_model, prediction_input, labels):
    embeddings = eval_model.predict(prediction_input, batch_size=600)
    tsne_projections = visual_utils.fit_transform_tsne(embeddings)
    tsne_figure = visual_utils.plot_tsne(tsne_projections, labels, label_name_list=dataset_full.info['session_list'])
    tsne_image = visual_utils.plot_to_image(tsne_figure)
    image = Image.fromarray(tsne_image[0])
    image.save(f"{working_directory}/results/{run_name}_tsne.png", format='PNG')
    return tsne_image

def fine_tune_evaluate(dataset_full, trained_model_save_path, args):

    all_devices = list(dataset_full.info['device_list'])

    if args.held_out is None:
        held_out_users = []
    elif args.held_out_num_groups is None:
        held_out_users = [dataset_full.info['user_list'][args.held_out]]
    else:
        held_out_users = get_group_held_out_users(dataset_full.info['user_list'], args.held_out, args.held_out_num_groups)


    input_shape = dataset_full.input_shape
    
    output_shape = len(dataset_full.info['session_list'])

    if args.multi_anchor:
        working_directory = os.path.join(args.working_directory, 'multi_anchor', args.multi_anchor_training_mode, args.exp_name, args.training_mode)
    elif args.multi_anchor_batches:
        working_directory = os.path.join(args.working_directory, 'multi_anchor_batches', args.multi_anchor_training_mode, args.exp_name, args.training_mode)
    else:
        working_directory = os.path.join(args.working_directory, args.train_device, args.exp_name, args.training_mode)
    
    if args.adaptation_layer:
        working_directory = os.path.join(working_directory, args.adaptation_architecture)
    
    working_directory = os.path.join(working_directory, args.eval_mode)

    if args.generalizability:
        working_directory = os.path.join(working_directory, 'generalizability')
    else: 
        working_directory = os.path.join(working_directory, 'adaptability')

    if not os.path.exists(working_directory):
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'models/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'logs/'), exist_ok=True)
        os.makedirs(os.path.join(working_directory, 'results/'), exist_ok=True)

    """if args.trained_model_path is None:
        if args.model_arch == '1d_conv':
            _, last_freeze_layer = simclr_models.create_base_model(input_shape, len(args.) ,model_name="1d_conv")
    else:
        # base_model = tf.keras.models.load_model(args.trained_model_path)"""
    last_freeze_layer = args.trained_model_last_freeze_layer

    if args.eval_mode != 'none':

        if args.training_mode != 'none':
            eval_model_path = trained_model_save_path
        else:
            eval_model_path = args.trained_model_path
        

        if args.eval_mode == 'base_model':

            total_epochs = args.fine_tune_epochs
            batch_size = args.fine_tune_batch_size

            training_set_device_X = np.concatenate([dataset_full.device_user_ds[args.fine_tune_device][u][0] for u in sorted(dataset_full.device_user_ds[args.fine_tune_device].keys()) if u not in held_out_users], axis=0)
            
            training_set_device_y = np.concatenate([dataset_full.device_user_ds[args.fine_tune_device][u][1] for u in sorted(dataset_full.device_user_ds[args.fine_tune_device].keys()) if u not in held_out_users], axis=0)

            training_full_size = training_set_device_y.shape[0]
            train_samples_count = int(training_full_size * args.fine_tune_take)
            shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
            training_set_device_X = training_set_device_X[shuffle_indices][:train_samples_count]
            training_set_device_y = tf.keras.utils.to_categorical(training_set_device_y[shuffle_indices][:train_samples_count], num_classes=output_shape)

            # %%
            ## Full HAR Model

            tag = args.eval_mode + "_eval"

            eval_model = tf.keras.models.load_model(eval_model_path, custom_objects={"AdaptationLayer": AdaptationLayer})
        
            # lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4, decay_steps=1000)
            optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=1e-4)
            full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(eval_model, output_shape, optimizer_fine_tune, model_name="TPN", intermediate_layer=-1, last_freeze_layer=last_freeze_layer)

            full_eval_best_model_file_name = f"{working_directory}/models/{args.run_name}_{tag}.keras"
            best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
                monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
            )

            scores = []
            eval_devices = copy.deepcopy(args.multi_eval_list)
            if args.eval_device in eval_devices:
                eval_devices.remove(args.eval_device)
            eval_devices = eval_devices + [args.eval_device] # evaluating eval_device last to return correct results
            adaptation_device_id = get_adaptation_group(args.adaptation_groups, args.fine_tune_device)[0]

            if args.adaptation_layer:
                adapt_layers = []
                for layer in full_evaluation_model.layers:
                    if isinstance(layer, AdaptationLayer):
                        adapt_layers.append(layer)
                        layer.adapt_mode = True # set all adapt modes to true for fine-tuning
                
                if args.generalizability and (args.fine_tune_device == args.eval_device) and (len(args.adaptation_groups[adaptation_device_id]) == 1): # fine-tune device was not seen in training and does not share an adaptation layer with any other devices
                    # calculate closest device to use for adaptation layer selection
                    training_set_device = [(np.concatenate([dataset_full.device_user_ds[d][u][0] for u in dataset_full.device_user_ds[d] if u not in held_out_users], axis=0)) for d in all_devices]
                    training_set_stacked = np.stack(training_set_device, axis=0)
                    tf_train_contrast = BatchedRandomisedDataset(training_set_stacked, batch_size, randomised=False, axis=1, name="distances")
                    positive_indices, negative_indices, distances = get_pos_neg_apriori(tf_train_contrast, all_devices, anchor_index=adaptation_device_id, strategy=args.device_selection_strategy)
                    closest_neighbor = positive_indices[0]
                    for layer in adapt_layers:
                        layer.sub_layers[adaptation_device_id].set_weights(copy.deepcopy(layer.sub_layers[closest_neighbor].get_weights()))
                
                adaptation_selections  = np.full((training_set_device_X.shape[0],), adaptation_device_id , dtype=np.int32)
                training_input = {"input": np.array(training_set_device_X), "layer_selection": adaptation_selections}
            else:
                training_input = np.array(training_set_device_X)

            training_history = full_evaluation_model.fit(
                x = training_input,
                y = training_set_device_y,
                validation_split = 0.2,
                batch_size = batch_size,
                validation_batch_size = batch_size,
                shuffle = False,
                epochs=total_epochs,
                callbacks=[best_model_callback],# myCallback()],
                #validation_data=val_split
                verbose=2
            )

            full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name, custom_objects={"AdaptationLayer": AdaptationLayer})

            scores = []
            for device in eval_devices:

                np_test_x = np.concatenate([dataset_full.device_user_ds[device][u][0] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = np.concatenate([dataset_full.device_user_ds[device][u][1] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = tf.keras.utils.to_categorical(np_test_y, num_classes=output_shape)
                np_test = (np_test_x, np_test_y)
                
                if args.adaptation_layer:
                    prediction_input = [np_test[0], np.full((np_test[0].shape[0],), adaptation_device_id , dtype=np.int32)]
                else:
                    prediction_input = np_test[0]

                results_lowest_loss = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(prediction_input), np_test[1], return_dict=True)
                results_last_epoch = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(prediction_input), np_test[1], return_dict=True)

                run_name = args.run_name
                run_name += f"-{args.eval_mode}-{args.fine_tune_device}-{device}"
                with open(f"{working_directory}/results/{run_name}.txt", 'a') as f:
                    f.write(f"Model path: {full_eval_best_model_file_name}\n")
                    f.write(f"Evaluation Mode: {args.eval_mode}, Fine Tune Device: {args.fine_tune_device}, Evalutaion Device: {device}\n")
                    f.write("====== Full Evaluation ======\n")
                    f.write("Model with lowest validation Loss:\n")
                    f.write(str(results_lowest_loss) + "\n")
                    f.write("Model in last epoch:\n")
                    f.write(str(results_last_epoch) + "\n")

                print("Model with lowest validation Loss:")
                print(results_lowest_loss)
                print("Model in last epoch")
                print(results_last_epoch)

                scores.append((device, round(max(results_lowest_loss['F1 Macro'], results_last_epoch['F1 Macro']),3))) # combine multiple results in one file
                if args.output_tsne:
                    tsne_image = output_tsne_image(working_directory, run_name, eval_model, prediction_input, np_test[1])


        elif args.eval_mode == 'full_model':  # This is to be run only with supervised setting
            assert args.training_mode == 'supervised', "args.eval_model = full_model can only be run with supervised training, use args.eval_model = base_model"

            full_eval_best_model = tf.keras.models.load_model(eval_model_path, custom_objects={"AdaptationLayer": AdaptationLayer})

            scores = []
            eval_devices = copy.deepcopy(args.multi_eval_list)
            if args.eval_device in eval_devices:
                eval_devices.remove(args.eval_device)
            eval_devices = eval_devices + [args.eval_device] # evaluating eval_device last to return correct results


            for device in eval_devices:

                np_test_x = np.concatenate([dataset_full.device_user_ds[device][u][0] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = np.concatenate([dataset_full.device_user_ds[device][u][1] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = tf.keras.utils.to_categorical(np_test_y, num_classes=output_shape)
                np_test = (np_test_x, np_test_y)
                
                prediction_input = np_test[0]

                results_lowest_loss = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(prediction_input), np_test[1], return_dict=True)

                run_name = args.run_name
                run_name += f"-{args.eval_mode}-{args.fine_tune_device}-{device}"
                with open(f"{working_directory}/results/{run_name}.txt", 'a') as f:
                    f.write(f"Model path: {eval_model_path}\n")
                    f.write(f"Evaluation Mode: {args.eval_mode}, Fine Tune Device: {args.fine_tune_device}, Evalutaion Device: {device}\n")
                    f.write("====== Full Evaluation ======\n")
                    f.write("Model with lowest validation Loss:\n")
                    f.write(str(results_lowest_loss) + "\n")

                print("Model with lowest validation Loss:")
                print(results_lowest_loss)

                scores.append((device, round(max(results_lowest_loss['F1 Macro'], results_last_epoch['F1 Macro']),3))) # combine multiple results in one file
                if args.output_tsne:
                    tsne_image = output_tsne_image(working_directory, run_name, eval_model, prediction_input, np_test[1])

        elif args.eval_mode == 'multi_ft':

            total_epochs = args.fine_tune_epochs
            batch_size = args.fine_tune_batch_size

            eval_model = tf.keras.models.load_model(eval_model_path, custom_objects={"AdaptationLayer": AdaptationLayer})

            # %%
            ## Full HAR Model

            tag = args.eval_mode + "_eval"

            
            # lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-4, decay_steps=1000)
            optimizer_fine_tune = tf.keras.optimizers.Adam(learning_rate=1e-4)

            if args.adaptation_classifier:
                #assert args.adaptation_layer, "args.adapt_classifier can only be run with adaptation layers enabled, use --adaptation_layer and choose an --adaptation_achritectore containing atleast one adaptation layer." 
                full_evaluation_model = simclr_models.create_multi_classification_model_from_base_model(base_model=eval_model, num_sub_layers=len(args.adaptation_groups), output_shape=output_shape, optimizer=optimizer_fine_tune, model_name="TPN", intermediate_layer=-1, last_freeze_layer=last_freeze_layer)
            else:
                full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(eval_model, output_shape, optimizer_fine_tune, model_name="TPN", intermediate_layer=-1, last_freeze_layer=last_freeze_layer) 
            

            fine_tune_devices = args.multi_finetune_list
            if not fine_tune_devices:
                fine_tune_devices = list(dataset_full.info['device_list'])


            training_set_X = []
            training_set_Y = []
            adaptation_selections = []
            for ft_device in fine_tune_devices:

                adaptation_device_id = get_adaptation_group(args.adaptation_groups, ft_device)[0]
                adapt_layers = []
                for layer in full_evaluation_model.layers:
                    if isinstance(layer, AdaptationLayer):
                        adapt_layers.append(layer)
                        layer.adapt_mode = True # set all adapt modes to true for fine-tuning
                    
                if args.generalizability and (ft_device == args.eval_device) and (len(args.adaptation_groups[adaptation_device_id]) == 1): # fine-tune device was not seen in training and does not share an adaptation layer with any other devices
                    # calculate closest device to use for adaptation layer selection
                    training_set_device = [(np.concatenate([dataset_full.device_user_ds[d][u][0] for u in dataset_full.device_user_ds[d] if u not in held_out_users], axis=0)) for d in all_devices]
                    training_set_stacked = np.stack(training_set_device, axis=0)
                    tf_train_contrast = BatchedRandomisedDataset(training_set_stacked, batch_size, randomised=False, axis=1, name="distances")
                    positive_indices, negative_indices, distances = get_pos_neg_apriori(tf_train_contrast, all_devices, anchor_index=adaptation_device_id, strategy=args.device_selection_strategy)
                    closest_neighbor = positive_indices[0]
                    for layer in adapt_layers:
                        layer.sub_layers[adaptation_device_id].set_weights(copy.deepcopy(layer.sub_layers[closest_neighbor].get_weights()))

                training_set_device_X = np.concatenate([dataset_full.device_user_ds[ft_device][u][0] for u in sorted(dataset_full.device_user_ds[ft_device].keys()) if u not in held_out_users], axis=0)
                training_set_device_y = np.concatenate([dataset_full.device_user_ds[ft_device][u][1] for u in sorted(dataset_full.device_user_ds[ft_device].keys()) if u not in held_out_users], axis=0)

                training_set_X.append(training_set_device_X)
                training_set_Y.append(training_set_device_y)

                adaptation_selections.append(np.full((training_set_device_X.shape[0],), adaptation_device_id , dtype=np.int32))



            training_set_X = np.concatenate([du for du in training_set_X], axis=0)
            training_set_Y = np.concatenate([du for du in training_set_Y], axis=0)
            adaptation_selections = np.concatenate([du for du in adaptation_selections], axis=0)


            training_full_size = training_set_Y.shape[0]
            train_samples_count = int(training_full_size * args.fine_tune_take) #// len(fine_tune_devices) # normalize for the number of devices
            shuffle_indices = get_random_shuffle_indices(training_full_size, seed=42)
            training_set_X = training_set_X[shuffle_indices][:train_samples_count]
            training_set_Y = tf.keras.utils.to_categorical(training_set_Y[shuffle_indices][:train_samples_count], num_classes=output_shape)
            adaptation_selections = adaptation_selections[shuffle_indices][:train_samples_count]

            full_eval_best_model_file_name = f"{working_directory}/models/{args.run_name}_{tag}.keras"
            best_model_callback = tf.keras.callbacks.ModelCheckpoint(full_eval_best_model_file_name,
                monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
            )

            if args.adaptation_layer or args.adaptation_classifier:
                training_input = {"input": np.array(training_set_X), "layer_selection": np.array(adaptation_selections, dtype=np.int32)}
            else:
                training_input = np.array(training_set_X)          

            training_history = full_evaluation_model.fit(
                x = training_input,
                y = training_set_Y,
                validation_split = 0.2,
                batch_size = batch_size,
                validation_batch_size = batch_size,
                shuffle = False,
                epochs=total_epochs,
                callbacks=[best_model_callback],# myCallback()],
                #validation_data=val_split
                verbose=2
            )

            full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name, custom_objects={"AdaptationLayer": AdaptationLayer})

            eval_devices = copy.deepcopy(args.multi_eval_list)
            if args.eval_device in eval_devices:
                eval_devices.remove(args.eval_device)
            eval_devices = eval_devices + [args.eval_device] # evaluatiing eval_device last to return correct results

            scores = []
            for device in eval_devices:

                adaptation_device_id = get_adaptation_group(args.adaptation_groups, device)[0]

                np_test_x = np.concatenate([dataset_full.device_user_ds[device][u][0] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = np.concatenate([dataset_full.device_user_ds[device][u][1] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = tf.keras.utils.to_categorical(np_test_y, num_classes=output_shape)
                np_test = (np_test_x, np_test_y)

                if args.adaptation_layer or args.adaptation_classifier:
                    prediction_input = [np_test[0], np.full((np_test[0].shape[0],), adaptation_device_id , dtype=np.int32)]
                else:
                    prediction_input = np_test[0]

                results_lowest_loss = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(prediction_input), np_test[1], return_dict=True)
                results_last_epoch = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(prediction_input), np_test[1], return_dict=True)

                run_name = args.run_name
                run_name += f"-{args.eval_mode}-{args.eval_device}"
                with open(f"{working_directory}/results/{run_name}.txt", 'a') as f:
                    f.write(f"Model path: {full_eval_best_model_file_name}\n")
                    f.write(f"Evaluation Mode: {args.eval_mode}, Multi Anchor Devices: {args.multi_anchor_list}, Evalutaion Device: {device}\n")
                    f.write("====== Full Evaluation ======\n")
                    f.write("Model with lowest validation Loss:\n")
                    f.write(str(results_lowest_loss) + "\n")
                    f.write("Model in last epoch:\n")
                    f.write(str(results_last_epoch) + "\n")

                print("Model with lowest validation Loss:")
                print(results_lowest_loss)
                print("Model in last epoch")
                print(results_last_epoch)

                scores.append((device, round(max(results_lowest_loss['F1 Macro'], results_last_epoch['F1 Macro']),3))) # combine multiple results in one file
                if args.output_tsne:
                    tsne_image = output_tsne_image(working_directory, run_name, eval_model, prediction_input, np_test[1])
        
        elif args.eval_mode == 'no_ft':
            eval_model = tf.keras.models.load_model(eval_model_path, custom_objects={"AdaptationLayer": AdaptationLayer})

            fine_tune_devices = args.multi_eval_list
            if args.generalizability:
                fine_tune_devices.remove(args.eval_device)

            
            full_evaluation_model = simclr_models.create_linear_model_from_base_model(eval_model, output_shape, intermediate_layer=-1)

            full_eval_best_model = full_evaluation_model

            if args.multi_eval:
                eval_devices = copy.deepcopy(args.multi_eval_list)
                if args.eval_device in eval_devices:
                    eval_devices.remove(args.eval_device)
                eval_devices = eval_devices + [args.eval_device] # evaluatiing eval_device last to return correct results
            else:
                eval_devices = [args.eval_device]
            scores = []
            for device in eval_devices:

                np_test_x = np.concatenate([dataset_full.device_user_ds[device][u][0] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = np.concatenate([dataset_full.device_user_ds[device][u][1] for u in sorted(dataset_full.device_user_ds[device].keys()) if (args.held_out is None or u in held_out_users)], axis=0)
                np_test_y = tf.keras.utils.to_categorical(np_test_y, num_classes=output_shape)
                np_test = (np_test_x, np_test_y)

                results_lowest_loss = simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
                results_last_epoch = simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True)

                run_name = args.run_name
                run_name += f"-{args.eval_mode}-{args.eval_device}"
                with open(f"{working_directory}/results/{run_name}.txt", 'a') as f:
                    f.write(f"Model path: {args.trained_model_path}\n")
                    f.write(f"Evaluation Mode: {args.eval_mode}, Multi Anchor Devices: {args.multi_anchor_list}, Evalutaion Device: {device}\n")
                    f.write("====== Full Evaluation ======\n")
                    f.write("Model with lowest validation Loss:\n")
                    f.write(str(results_lowest_loss) + "\n")
                    f.write("Model in last epoch:\n")
                    f.write(str(results_last_epoch) + "\n")

                print("Model with lowest validation Loss:")
                print(results_lowest_loss)
                print("Model in last epoch")
                print(results_last_epoch)

                scores.append((device, round(max(results_lowest_loss['F1 Macro'], results_last_epoch['F1 Macro']),3))) # combine multiple results in one file
                if args.output_tsne:
                    tsne_image = output_tsne_image(working_directory, run_name, eval_model, prediction_input, np_test[1])
        

    if len(args.multi_eval_list) >= 2:
        values = [val for _, val in scores]
        avg = round(stats.mean(values), 3)  
        stdev = round(stats.stdev(values), 3)
        minimum = round(min(values), 3)
        adi = round(0.5*avg - 0.2*stdev + 0.3*minimum, 3)
        if len(eval_devices) > 1:
            # output multi evalutation file:
            run_name = args.run_name
            run_name += f"-{args.eval_mode}-{args.fine_tune_device}"
            with open(f"{working_directory}/results/{run_name}-multi_evaluation.txt", 'a') as f:
                f.write(f"Model path: {trained_model_save_path}\n")
                f.write("\n====== Args ======\n")
                f.write(str(args) + "\n")
                f.write(f"Training devices: {args.multi_anchor_list if (args.multi_anchor or args.multi_anchor_batches) else args.train_device}\n")
                if args.adaptation_layer:
                    f.write(f"Adaptation groups: {args.adaptation_groups}\n")
                    f.write(f"Adaptation classifier: {args.adaptation_classifier}\n")
                f.write(f"Fine-tuning devices: {args.multi_finetune_list}\n")
                f.write("====== Multi Evaluation ======\n")
                f.write("Evaluation devices:\n")
                for score in scores:
                    f.write(f"{score[0]}: {score[1]}" + "\n")
                f.write("====== Statistics ======\n")
                f.write(f"Average F1: {avg}\n")
                f.write(f"Std. deviation: {stdev}\n")
                f.write(f"Minimum: {minimum}\n")
                f.write(f"Adaptability Index: {adi}\n")

    if args.eval_mode == 'full_model':
        return results_lowest_loss['F1 Macro'], results_lowest_loss['F1 Weighted'], tsne_image 

    else: # eval_mode in {base_model, multi_ft, no_ft}
        if results_lowest_loss['F1 Macro'] >= results_last_epoch['F1 Macro']:
            return results_lowest_loss['F1 Macro'], results_lowest_loss['F1 Weighted'], tsne_image
        else:
            return results_last_epoch['F1 Macro'], results_last_epoch['F1 Weighted'], tsne_image    


if __name__ == '__main__':
    
    parser = get_parser()

    ## Prepare full dataset
    args = parser.parse_args()
    #print(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    os.environ['PYTHONHASHSEED']='42'
    os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS']='1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.random.set_seed(42)
    rng = np.random.default_rng(seed=42)

    run_id = str(datetime.datetime.now())[-6:] #random.randint(10**(6-1), (10**6)-1)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for sel_gpu in gpus:
                tf.config.experimental.set_memory_growth(sel_gpu, True)
        except RuntimeError as e:
            print(e)

    dataset_full = load_data.Data(args.dataset_path, args.dataset_name, load_path=None, held_out=args.held_out)

    # set defaults for train_device, fine_tune_device, eval_device, multi_eval_list and multi_finetune_list, multi_anchor_list

    if args.adaptation_layer:
        assert 'a' in list(args.adaptation_architecture.split('-')), "adaptation_layer is enabled. Please choose atleast one convolution layer to replace with an AdaptationLayer using the adaptation_architecture argument"
    if args.reg_loss:
        assert args.adaptation_layer, "Regulation loss enabled without using adaptation layers. Either use adaptation layer(s) or disable regulation loss"
    if args.train_device is None:
        if args.generalizability and args.eval_device == list(dataset_full.info['device_list'])[0]:
            args.train_device = list(dataset_full.info['device_list'])[1]
        else:
            args.train_device = list(dataset_full.info['device_list'])[0]
    if args.eval_device is None:
        args.eval_device = args.train_device
    if args.fine_tune_device is None:
        args.fine_tune_device = args.eval_device
    if not args.multi_eval_list:
        args.multi_eval_list = [args.eval_device]
    if not args.multi_finetune_list:
        args.multi_finetune_list = [args.fine_tune_device]
    if not args.multi_anchor_list:
        args.multi_anchor_list = [list(dataset_full.info['device_list'])] # set all devices as default list due to argparse parsing restirctions with 'append' action
    if not args.adaptation_groups:
        args.adaptation_groups = [[d] for d in list(dataset_full.info['device_list'])] # default to one adaptation layer per device in the dataset
    if not hasattr(args, 'run_name'):
        if args.multi_anchor or args.multi_anchor_batches:
            args.run_name = f"run-{run_id}-{args.training_mode}-{args.multi_anchor_list}-{args.multi_anchor_training_mode}"
        else:
            args.run_name = f"run-{run_id}-{args.training_mode}-{args.train_device}"
        if args.generalizability:
            args.run_name += "-generalizability"


    if args.multi_anchor:
        multi_anchor_list = copy.deepcopy(args.multi_anchor_list)
        args_copy = copy.deepcopy(args)
        epochs_total = args.training_epochs
        if args.generalizability: 
            # remove evaluation device from each sub group of multi anchor list
            for group in multi_anchor_list:
                if args_copy.eval_device in group:
                    group.remove(args_copy.eval_device)
                if not group:
                    multi_anchor_list.remove(group)
        num_groups = len(multi_anchor_list)

        if args.multi_anchor_training_mode == 'divided':
            cycle_epochs = epochs_total // num_groups
            for group in multi_anchor_list:
                args_copy.multi_anchor_list = group
                args_copy.train_device = args_copy.multi_anchor_list[0]
                args_copy.training_epochs = cycle_epochs

                print("Training", group, "for ", cycle_epochs, "epochs")

                trained_model_save_path = train(dataset_full, args_copy)
                args_copy.trained_model_path = trained_model_save_path

        args.trained_model_path = args_copy.trained_model_path
        args.trained_model_last_freeze_layer = args_copy.trained_model_last_freeze_layer
    else: # single anchor training
        multi_anchor_list = copy.deepcopy(args.multi_anchor_list)
        args.multi_anchor_list = multi_anchor_list[0]
        trained_model_save_path = train(dataset_full, args)

    if args.eval_mode == 'multi_ft' and not args.multi_finetune_list:
        args.multi_finetune_list = list(dataset_full.info['device_list'])

    if args.multi_eval: 
        # evaluate the model using both 'base model' and 'multi_ft' evaluation  
        args.eval_mode = 'multi_ft'
        multi_finetune_list = copy.deepcopy(args.multi_finetune_list)
        fine_tune_evaluate(dataset_full, trained_model_save_path, args)

        args.eval_mode = 'base_model'
        if args.generalizability:
            multi_finetune_list = [args.eval_device]
        for e_device in multi_finetune_list:
            args.fine_tune_device = e_device
            fine_tune_evaluate(dataset_full, trained_model_save_path, args)

      
    else: # evaluate model using a single evalutation method
        if args.multi_finetune_list and (args.eval_mode == 'base_model'):
            if args.generalizability:
                multi_finetune_list = [args.eval_device]
            else:
                multi_finetune_list = copy.deepcopy(args.multi_finetune_list)
            for e_device in multi_finetune_list:
                args.fine_tune_device = e_device
                fine_tune_evaluate(dataset_full, trained_model_save_path, args)
        else:
            fine_tune_evaluate(dataset_full, trained_model_save_path, args)
            
