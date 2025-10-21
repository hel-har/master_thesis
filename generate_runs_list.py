import datetime
import pickle 
import os

model_name = "collossl"
dataset_path = "collossl/data/gsl/" # change path to dataset files
root_path = "collossl/data/gsl/master_tests/"
if not os.path.exists(root_path):
    os.makedirs(root_path)

list_dir_path = "run_lists/collossl/"
if not os.path.exists(list_dir_path):
    os.makedirs(list_dir_path)

datasets=["realworld-3.0-0.0.dat","pamap2_test.dat"]

dataset_devices = dict()
train_devices = dict()
eval_devices = dict()
for dataset in datasets:
    with open(os.path.join(dataset_path, dataset), 'rb') as f:
        data = pickle.load(f)

    dataset_devices[dataset] = list(data[0]['device_list'])
    train_devices[dataset] = dataset_devices[dataset]
    eval_devices[dataset] = dataset_devices[dataset]

training_modes = ["divided", "cycle", "randomized"]

def stringify(list):#
    string = ""
    for item in list:
        string += item + " "
    return string[:-1]

### Baseline: Supervised - Adaptability + Generalizability (Realworld + Pamap2)

file_name = "supervised-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

# Supervised runs with single anchor training and full_model evaluataion
working_directory = os.path.join(root_path, "runs-baseline/supervised/single_device_training")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
with open(file_path,'w+') as f:
    for dataset in datasets:
        for t_device in train_devices[dataset]:
            out_adapt = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode supervised \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --eval_mode=full_model \
                        --train_device {t_device} \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        "
            f.write(out_adapt + "\n")

# Supervised runs with single anchor training and multi evaluation (base_model + multi_ft)
with open(file_path,'a+') as f:
    for dataset in datasets:
        for t_device in train_devices[dataset]:
            out_adapt = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode supervised \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --multi_eval \
                        --train_device {t_device} \
                        --multi_finetune_list {stringify(eval_devices[dataset])} \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        "
            f.write(out_adapt + "\n")

# Supervised runs with multi anchor (pre-)training and full_model evaluation
working_directory = os.path.join(root_path, "runs-baseline/supervised/multi_device_training")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
with open(file_path,'a+') as f:
    for dataset in datasets:
        out_adapt = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode supervised \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --baseline=supervised_all_devices \
                        --eval_mode=full_model \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        "
        f.write(out_adapt + "\n")
        for e_device in eval_devices[dataset]:
            out_general = out_adapt + f"--eval_device {e_device} \
                                        --generalizability"
            
            f.write(out_general + "\n")


# Supervised runs with multi anchor (pre-)training and multi evaluation (base_model + multi_ft)
with open(file_path,'a+') as f:
    for dataset in datasets:
        out_adapt = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode supervised \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --baseline=supervised_all_devices \
                        --multi_eval \
                        --multi_finetune_list {stringify(eval_devices[dataset])} \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        "
        f.write(out_adapt + "\n")
        for e_device in eval_devices[dataset]:
            out_general = out_adapt + f" \
                                        --fine_tune_device {e_device} \
                                        --eval_device {e_device} \
                                        --generalizability \
                                        "
            
            f.write(out_general + "\n")


### Baseline: ColloSSL - Adaptability + Generalizability (Realworld + Pamap2)

file_name = "collossl-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

# Base ColloSSL runs with single anchor training and multi evaluation (base_model + multi_ft)
working_directory = os.path.join(root_path, "runs-baseline/collossl/single_device_training")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
with open(file_path,'w+') as f:
    for dataset in datasets:
        for t_device in train_devices[dataset]:
            out_adapt = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode multi \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --multi_eval \
                        --train_device {t_device} \
                        --multi_finetune_list {stringify(eval_devices[dataset])} \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        "
            f.write(out_adapt + "\n")

# Base ColloSSL runs with multi anchor training (within each batch) and multi evaluation (base_model + multi_ft)
working_directory = os.path.join(root_path, "runs-baseline/collossl/multi_device_training_batches")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
with open(file_path,'a+') as f:
    for dataset in datasets:
        for t_mode in training_modes:
            out_adapt = f"python3 collossl/contrastive_training.py \
                            --exp_name={dataset} \
                            --working_directory {working_directory} \
                            --dataset_path {dataset_path} \
                            --gpu_device=0 \
                            --training_mode multi \
                            --training_epochs=100 \
                            --multi_sampling_mode=unsync_neg \
                            --held_out=0 \
                            --held_out_num_groups=5 \
                            --device_selection_strategy=closest_pos_all_neg \
                            --weighted_collossl \
                            --training_batch_size=512 \
                            --positive_devices \
                            --negative_devices \
                            --contrastive_temperature=0.05 \
                            --fine_tune_epochs=100 \
                            --learning_rate=1e-4 \
                            --dataset_name={dataset} \
                            --neg_sample_size=1 \
                            --dynamic_device_selection=1 \
                            --multi_anchor_batches \
                            --multi_anchor_training_mode {t_mode} \
                            --multi_eval \
                            --multi_finetune_list {stringify(eval_devices[dataset])} \
                            --multi_eval_list {stringify(eval_devices[dataset])} \
                            "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                out_general = out_adapt + f" \
                                            --fine_tune_device {e_device} \
                                            --eval_device {e_device} \
                                            --generalizability \
                                            "
                
                f.write(out_general + "\n")


### Multi-Anchor: ColloSSL + multi_anchor training (switching anchor between epochs) and multi evaluation (Realworld + Pamap2)

file_name = "multi_anchor-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

working_directory = os.path.join(root_path, "runs-multi_anchor/collossl/multi_device_training_epochs")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
with open(file_path,'w+') as f:
    for dataset in datasets:
        for t_mode in training_modes:
            out_adapt = f"python3 collossl/contrastive_training.py \
                            --exp_name={dataset} \
                            --working_directory {working_directory} \
                            --dataset_path {dataset_path} \
                            --gpu_device=0 \
                            --training_mode multi \
                            --training_epochs=100 \
                            --multi_sampling_mode=unsync_neg \
                            --held_out=0 \
                            --held_out_num_groups=5 \
                            --device_selection_strategy=closest_pos_all_neg \
                            --weighted_collossl \
                            --training_batch_size=512 \
                            --positive_devices \
                            --negative_devices \
                            --contrastive_temperature=0.05 \
                            --fine_tune_epochs=100 \
                            --learning_rate=1e-4 \
                            --dataset_name={dataset} \
                            --neg_sample_size=1 \
                            --dynamic_device_selection=1 \
                            --multi_anchor \
                            --multi_anchor_training_mode {t_mode} \
                            --multi_anchor_list {stringify(train_devices[dataset]).replace(' ', ' --multi_anchor_list ')} \
                            --multi_eval \
                            --multi_finetune_list {stringify(eval_devices[dataset])} \
                            --multi_eval_list {stringify(eval_devices[dataset])} \
                            "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                out_general = out_adapt + f" \
                                            --fine_tune_device {e_device} \
                                            --eval_device {e_device} \
                                            --generalizability \
                                            "
                
                f.write(out_general + "\n")


### Adaptation-Layer: ColloSSL + adaptation layer (one conv. layer is replaced by multiple conv. layers that are trained simultaneously on different devices) using 3 different architectures (Realworld + Pamap2)

file_name = "multi_anchor-adaptation_layer-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

working_directory = os.path.join(root_path, "runs-multi_anchor-adaptation_layer")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

architectures = ['a-c-c', 'c-a-c', 'c-c-a']

with open(file_path,'w+') as f:
    for dataset in datasets:
        for architecture in architectures:
            out_adapt = f"python3 collossl/contrastive_training.py \
                            --exp_name={dataset} \
                            --working_directory {working_directory} \
                            --dataset_path {dataset_path} \
                            --gpu_device=0 \
                            --training_mode multi \
                            --training_epochs=100 \
                            --multi_sampling_mode=unsync_neg \
                            --held_out=0 \
                            --held_out_num_groups=5 \
                            --device_selection_strategy=closest_pos_all_neg \
                            --weighted_collossl \
                            --training_batch_size=512 \
                            --positive_devices \
                            --negative_devices \
                            --contrastive_temperature=0.05 \
                            --fine_tune_epochs=100 \
                            --learning_rate=1e-4 \
                            --dataset_name={dataset} \
                            --neg_sample_size=1 \
                            --dynamic_device_selection=1 \
                            --adaptation_layer \
                            --adaptation_architecture={architecture} \
                            --multi_anchor \
                            --multi_anchor_batches \
                            --multi_anchor_training_mode divided \
                            --multi_eval \
                            --multi_finetune_list {stringify(eval_devices[dataset])} \
                            --multi_eval_list {stringify(eval_devices[dataset])} \
                            --multi_anchor_list {stringify(train_devices[dataset])} \
                            "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                out_general = out_adapt + f" \
                                            --fine_tune_device {e_device} \
                                            --eval_device {e_device} \
                                            --generalizability \
                                            "
                
                f.write(out_general + "\n")

### Adaptation-Classifier: ColloSSL + adaptation layer +  adaptation classifier, using 3 different architectures (Realworld + Pamap2)

file_name = "multi_anchor-adaptation_classifier-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

working_directory = os.path.join(root_path, "runs-multi_anchor-adaptation_classifier")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

architectures = ['a-c-c', 'c-a-c', 'c-c-a']

with open(file_path,'w+') as f:
    for dataset in datasets:
        for architecture in architectures:
            out_adapt = f"python3 collossl/contrastive_training.py \
                            --exp_name={dataset} \
                            --working_directory {working_directory} \
                            --dataset_path {dataset_path} \
                            --gpu_device=0 \
                            --training_mode multi \
                            --training_epochs=100 \
                            --multi_sampling_mode=unsync_neg \
                            --held_out=0 \
                            --held_out_num_groups=5 \
                            --device_selection_strategy=closest_pos_all_neg \
                            --weighted_collossl \
                            --training_batch_size=512 \
                            --positive_devices \
                            --negative_devices \
                            --contrastive_temperature=0.05 \
                            --fine_tune_epochs=100 \
                            --learning_rate=1e-4 \
                            --dataset_name={dataset} \
                            --neg_sample_size=1 \
                            --dynamic_device_selection=1 \
                            --adaptation_layer \
                            --adaptation_architecture={architecture} \
                            --multi_anchor \
                            --multi_anchor_batches \
                            --multi_anchor_training_mode divided \
                            --multi_eval \
                            --multi_finetune_list {stringify(eval_devices[dataset])} \
                            --multi_eval_list {stringify(eval_devices[dataset])} \
                            --multi_anchor_list {stringify(train_devices[dataset])} \
                            --adaptation_classifier \
                            "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                out_general = out_adapt + f" \
                                            --fine_tune_device {e_device} \
                                            --eval_device {e_device} \
                                            --generalizability \
                                            "
                
                f.write(out_general + "\n")


### Adaptation-Classifier: ColloSSL + adaptation layer +  regulation loss, using 3 different architectures (Realworld + Pamap2)

file_name = "multi_anchor-reg_loss-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

working_directory = os.path.join(root_path, "runs-multi_anchor-reg_loss")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

architectures = ['a-c-c', 'c-a-c', 'c-c-a']

with open(file_path,'w+') as f:
    for dataset in datasets:
        for architecture in architectures:
            out_adapt = f"python3 collossl/contrastive_training.py \
                            --exp_name={dataset} \
                            --working_directory {working_directory} \
                            --dataset_path {dataset_path} \
                            --gpu_device=0 \
                            --training_mode multi \
                            --training_epochs=100 \
                            --multi_sampling_mode=unsync_neg \
                            --held_out=0 \
                            --held_out_num_groups=5 \
                            --device_selection_strategy=closest_pos_all_neg \
                            --weighted_collossl \
                            --training_batch_size=512 \
                            --positive_devices \
                            --negative_devices \
                            --contrastive_temperature=0.05 \
                            --fine_tune_epochs=100 \
                            --learning_rate=1e-4 \
                            --dataset_name={dataset} \
                            --neg_sample_size=1 \
                            --dynamic_device_selection=1 \
                            --adaptation_layer \
                            --adaptation_architecture={architecture} \
                            --multi_anchor \
                            --multi_anchor_batches \
                            --multi_anchor_training_mode divided \
                            --eval_mode multi_ft \
                            --multi_finetune_list {stringify(eval_devices[dataset])} \
                            --multi_eval_list {stringify(eval_devices[dataset])} \
                            --multi_anchor_list {stringify(train_devices[dataset])} \
                            --reg_loss \
                            "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                out_general = out_adapt + f" \
                                            --fine_tune_device {e_device} \
                                            --eval_device {e_device} \
                                            --generalizability \
                                            "
                
                f.write(out_general + "\n")





### Groups:  ColloSSL runs with first half multi-anchor training and second half device specific training (Realworld + Pamap2)

file_name = "multi_anchor-groups-adaptability-generalizability"
file_path = os.path.join(list_dir_path, file_name)

# Runs without adaptation layer

working_directory = os.path.join(root_path, "runs-multi_anchor-groups/no_adaptation_layer")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

with open(file_path,'w+') as f:
    for dataset in datasets:
        out_base = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode multi \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --multi_anchor \
                        --multi_anchor_batches \
                        --multi_anchor_training_mode divided \
                        --multi_eval \
                        --multi_finetune_list {stringify(eval_devices[dataset])} \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        --multi_anchor_list {stringify(train_devices[dataset])} \
                        "
        for t_device in train_devices[dataset]:
            out_adapt = out_base + f" \
                                     --multi_anchor_list {t_device} \
                                    "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                if e_device != t_device:
                    out_general = out_adapt + f" \
                                                --fine_tune_device {e_device} \
                                                --eval_device {e_device} \
                                                --generalizability \
                                                "
                    
                    f.write(out_general + "\n")

# Runs with adaptation layer
working_directory = os.path.join(root_path, "runs-multi_anchor-groups/adaptation_layer")
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

with open(file_path,'a+') as f:
    for dataset in datasets:
        out_base = f"python3 collossl/contrastive_training.py \
                        --exp_name={dataset} \
                        --working_directory {working_directory} \
                        --dataset_path {dataset_path} \
                        --gpu_device=0 \
                        --training_mode multi \
                        --training_epochs=100 \
                        --multi_sampling_mode=unsync_neg \
                        --held_out=0 \
                        --held_out_num_groups=5 \
                        --device_selection_strategy=closest_pos_all_neg \
                        --weighted_collossl \
                        --training_batch_size=512 \
                        --positive_devices \
                        --negative_devices \
                        --contrastive_temperature=0.05 \
                        --fine_tune_epochs=100 \
                        --learning_rate=1e-4 \
                        --dataset_name={dataset} \
                        --neg_sample_size=1 \
                        --dynamic_device_selection=1 \
                        --adaptation_layer \
                        --multi_anchor \
                        --multi_anchor_batches \
                        --multi_anchor_training_mode divided \
                        --multi_eval \
                        --multi_finetune_list {stringify(eval_devices[dataset])} \
                        --multi_eval_list {stringify(eval_devices[dataset])} \
                        --multi_anchor_list {stringify(train_devices[dataset])} \
                        "
        for t_device in train_devices[dataset]:
            out_adapt = out_base + f" \
                                     --multi_anchor_list {t_device} \
                                    "
            f.write(out_adapt + "\n")
            for e_device in eval_devices[dataset]:
                if e_device != t_device:
                    out_general = out_adapt + f" \
                                                --fine_tune_device {e_device} \
                                                --eval_device {e_device} \
                                                --generalizability \
                                                "
                    
                    f.write(out_general + "\n")