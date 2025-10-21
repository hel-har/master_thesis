import os
import random
import argparse
import datetime
import numpy as np
import copy
import tensorflow as tf
from common_parser import get_parser
from contrastive_training import fine_tune_evaluate
import load_data

if __name__ == '__main__':
    
    parser = get_parser()

    ## Prepare full dataset
    args = parser.parse_args()

    if args.eval_device is None:
        args.eval_device = args.train_device
    if args.fine_tune_device is None:
        args.fine_tune_device = args.train_device
    print(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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
    
    # train_and_evaluate(tf_dataset_full, args)
    trained_model_save_path = args.trained_model_path

    # Set default multi eval list to all devices of the used dataset
    if args.eval_mode == 'multi_ft' and not args.multi_finetune_list:
        args.multi_finetune_list = list(dataset_full.info['device_list'])

    if args.multi_eval: # evaluate the model using both 'base model' and 'multi_ft' evaluation
        
        args.eval_mode = 'multi_ft'
        multi_finetune_list = copy.deepcopy(args.multi_finetune_list)
        fine_tune_evaluate(dataset_full, trained_model_save_path, args)

        args.eval_mode = 'base_model'
        if args.generalizability:
            multi_finetune_list = [args.eval_device]
        for e_device in multi_finetune_list:
            args.fine_tune_device = e_device
            fine_tune_evaluate(dataset_full, trained_model_save_path, args)

        
    else:
        if args.multi_finetune_list and (args.eval_mode == 'base_model'):
            multi_finetune_list = copy.deepcopy(args.multi_finetune_list)
            for e_device in multi_finetune_list:
                args.fine_tune_device = e_device
                fine_tune_evaluate(dataset_full, trained_model_save_path, args)
        else:
            fine_tune_evaluate(dataset_full, trained_model_save_path, args)