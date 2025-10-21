import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description='Inputs to data loading script')
    parser.add_argument('--generalizability', default=False, action='store_true',
                        help='training option for differing training and eval devices, eval_device is removed from training')
    parser.add_argument('--dataset_path', default='/mnt/data/gsl/',
                        help='path of the dataset .dat file')
    parser.add_argument('--dataset_name', default='realworld-3.0-0.0.dat', choices=[
                        'realworld-3.0-0.0.dat', 'opportunity-1.0-0.0.dat', 'pamap2-2.0-1.0-v2.dat', 'pamap2adl-2.0-0.0.dat', 'pamap2_test.dat', 'hhar-2.0-0.5.dat', 'opportunity-1.6-0.0.dat', 'opportunity-2.0-0.0.dat'], help='name of dataset file')
    parser.add_argument('--load_path', default=None,
                        help='path of the dataset TFrecords files')

    parser.add_argument('--trained_model_path', default=None, type=str,
                        help='the path to the model to be loaded for training and/or evaluation. If None, a new model will be created')
    parser.add_argument('--trained_model_last_freeze_layer', default=12, type=int,
                        help='the last layer to be frozen for HAR model evaluation ')

    parser.add_argument('--working_directory', default='/mnt/data/gsl/runs/',
                        help='the directory where the trained models and results are saved')
    parser.add_argument('--exp_name', required=True,
                        help='the sub-directory where the trained models and results are saved')

    parser.add_argument('--baseline', default=None, type=str,
                        choices=[None, 'random', 'supervised_all_devices', 'multi_task_transform', 'en_co_training'],
                        help='select the baseline, None means no run, random means supervised fine-tuning of an initialised base model, supervised_all_devices means labelled data from all devices is used for pre-training')
    
    parser.add_argument('--training_mode', default='multi', type=str,
                        choices=['none', 'multi', 'supervised'],
                        help='the training setup, none means no training, single refers to single-device contrastive training, multi refers to \
                        multi-device contrastive training, supervised refers to end-to-end fully-supervised training')
    parser.add_argument('--train_device', default=None,
                        help='the device from which the data is used for training (including acting as the achor device for multi-device training)')
    parser.add_argument('--training_epochs', default=100, type=int,
                        help='number of epochs for training')
    parser.add_argument('--training_batch_size', default=512, type=int,
                        help='batch size for contrastive learning')
    parser.add_argument('--take', default=1.0, type=float,
                        help='percentage of training samples to take from the dataset. To use all samples, set 1.0 (100%)')
    parser.add_argument('--learning_rate_decay', default='cosine', type=str, choices=['cosine', 'none'],
                        help='the learning rate decay function')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='the initial learning rate during contrastive learning')
    parser.add_argument('--training_decay_steps', default=1000, type=int,
                        help='the total number of steps for Cosine Decay during contrastive learning')
    parser.add_argument('--contrastive_temperature', default=0.1, type=float,
                        help='the temperature hyperparameter for the NT-Xent Loss during contrastive learning')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd', 'rmsprop'],
                        help='optimizer to use for training')
    parser.add_argument('--model_arch', default='1d_conv',
                        help='model architecture to use')
    parser.add_argument('--data_aug', default='none', type=str, choices=['none', 'rotate', 'sensor_noise'],
                        help='data augmentations to use')

    parser.add_argument('--positive_devices', default=['forearm', 'upperarm', 'shin'], type=str, nargs='*',
                        help='list of positive devices for multi-device training')
    parser.add_argument('--negative_devices', default=['head', 'chest', 'waist'], type=str, nargs='*',
                        help='list of negative devices for multi-device training')
    parser.add_argument('--multi_sampling_mode', default='sync_all', type=str, choices=['sync_all', 'unsync_neg', 'unsync_all'],
                        help='the sampling method for multi-device training, sync_all refers to windows from all devices are synchronised, \
                        unsync_neg refers to the sampling for negative devices is unsynchronised within themselves as well as against anchor/positive devices, \
                        unsync_all refers to the sampling for all devices is unsynchronised.')
    parser.add_argument('--device_selection_metric', default='mmd_acc_norm', type=str, choices=['mmd_acc_norm','mmd_acc_per_channel'],
                        help='inter-device distance metric to follow for device selection')
    parser.add_argument('--dynamic_device_selection', default=0, type=int, choices=[0,1],
                        help='device selection dynamically in each batch otherwise device selection is done apriori')
    parser.add_argument('--neg_sample_size', default=1, type=int,
                        help='sample size of negative samples during contrastive training, if more than one then that number of negative samples are taken for each negative device')

    parser.add_argument('--eval_mode', default='base_model', type=str,
                        choices=['none', 'base_model', 'full_model', 'multi_ft', 'no_ft'],
                        help='the evaluation setup, none means no evaluation, base_model refers to base model (feature extractor) evaluation,\
                        in which a linear evaluation layer and full classification layers will be added for two evaluations,\
                        full_model refers to a direct evaluation of the model (works for classification models only)')
    parser.add_argument('--eval_device', default=None,
                        help='the device from which the data is used for evaluation, if is None, the train_device will be used')

    parser.add_argument('--fine_tune_device', default=None,
                        help='the device from which the data is used for fine-tuning, if is None, the train_device will be used')
    parser.add_argument('--fine_tune_take', default=1.0, type=float,
                        help='percentage of training samples to take from the dataset during fine-tuning. To use all samples, set 1.0 (100%)')
    parser.add_argument('--fine_tune_batch_size', default=32, type=int,
                        help='batch size for fine-tuning during evaluation')
    parser.add_argument('--fine_tune_epochs', default=100, type=int,
                        help='number of epochs for for fine-tuning during evaluation')

    parser.add_argument('--held_out', default=None, type=int,
                        help='the held out user or the held out group index during training and evaluation')
    parser.add_argument('--held_out_num_groups', default=None, type=int,
                        help='the number of groups for leave-one-group-out held-out evaluation. if it is None, leave-one-user-out is used instead')

    parser.add_argument('--output_tsne', default=0,  type=int,
                        help='to generate tsne plots or not. 0 for No, 1 for Yes')
    parser.add_argument('--gpu_device', default='0', type=str,
                        help='set the gpu device')

    parser.add_argument('--n_process_per_gpu', default=2, type=int,
                        help='number of processes per gpu (for multi-processing only)')

    parser.add_argument('--device_selection_strategy', default='closest_only', type=str, choices=['closest_only', 'hard_negative', 'closest_pos_all_neg','random_selection'],
                        help='selection strategy for selecting positive and negative devices as defined in device_selection_logic()')
    parser.add_argument('--weighted_collossl', default=False, action='store_true',
                        help='Whether to weight the group contrastive loss by the distance of each device from anchor')
    
    parser.add_argument('--multi_anchor', default=False, action='store_true',
                        help='Whether to contrastively train on multiple anchors or not, supersedes train_device argument')
    parser.add_argument('--multi_anchor_batches', default=False, action='store_true',
                        help='Whether to contrastively train on multiple anchors within a given epoch or not, supersedes train_device argument if applied')
    parser.add_argument('--multi_anchor_training_mode', default='divided', type=str, choices=['divided','cycle','randomized'],
                        help='Training mode for multi-anchor training. divided means equal training epochs for each anchor device using num_epochs/num_anchors epochs per device. cylce means cycling through anchor devices every num_cycle_epochs epochs. randomized chooses a random anchor device for every epoch')
    parser.add_argument('--multi_anchor_cycle', default=1, type=int,
                        help='Number of epochs/batches between switching anchor devices (for cycle and randomized modes)')
    parser.add_argument('--multi_anchor_list', default=[], type=str, nargs='*', action='append',
                        help='List of achor devices for multi-anchor training')

    parser.add_argument('--multi_eval', default=False, action='store_true',
                        help='Whether to evaluate the trained model on multiple evaluation devices')
    parser.add_argument('--multi_eval_list', default=[], type=str, nargs='*',
                        help='List of evaluation devices for multiple evaluations')
    parser.add_argument('--multi_finetune_list', default=[], type=str, nargs='*',
                        help='List of evaluation devices for multiple evaluations')
    parser.add_argument('--adaptation_layer', default=False, action='store_true',
                        help='Whether to replace convolution layers with device specific adaptation layers. Only usable with multi-anchor training.')
    parser.add_argument('--adaptation_classifier', default=False, action='store_true',
                        help='Whether to replace the fully connected layer within the classification head with adaptation layers. Only usable with adaptation layers enabled. Only works with multi fine-tuning.')
    parser.add_argument('--adaptation_architecture', default='c-c-c', type=str, #choices=['a-c-c', 'c-a-c', 'c-c-a'],
                        help='Which convolution layer to replace with the adaptation layer')
    parser.add_argument('--adaptation_groups', default=[], type=str, nargs='*', action='append',
                    help='Groups that are used to form adaptation layers (if enabled). Otherwise groups are ignored.')
    parser.add_argument('--reg_loss', default=False, action='store_true',
                        help='Whether to add a regulation loss term to penalize divergence of adaptation sub-layers. Only usable in conjunction with adaptation layers.')
    parser.add_argument('--adapt_mode_training', default=False, action='store_true',
                        help='Whether to use changes in the adapt mode of the adaptation layer during training. Only usabe in conmjunction with MULTIPLE adaptation layers. ')
 
    return parser
