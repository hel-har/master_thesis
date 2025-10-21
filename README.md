# Master Thesis - Helge Hartleb 
## [Towards Improving Generalizability By Using Self Supervised Learning Techniques for Deep Learning Models in Multi-Sensor-Based Human Activity Recognition](Master_Thesis.pdf)
This repository contains code regarding alterations made to the [ColloSSL framework](https://github.com/akhilmathurs/collossl) within the scope of the thesis. 

For a detailed description how to setup the framework, refer to the [original GitHub repository](https://github.com/akhilmathurs/collossl). 
Notes: This repository contains a [conda environment](environment.yml) used for all experiments during my research. Additionally, two python scripts are provided for the creation of a [custom version](create_pamap2.py) of the [PaMap2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) dataset and the generation of a list of [execution commands](generate_runs_list.py) that were used to run the experiments analyzed in my thesis.

# Alterations:
## Multi Fine-tuning
Fine-tune a model to multiple devices at once.
- **eval_mode=** *multi_ft*

## Multi Anchor Training
Train the model using multiple devices as anchors.
- **multi_anchor**, **multi_anchor_batches**: For epoch and batch dependent switching respectively.
- **multi_anchor_training_mode=** *{cycle, divided, randomized}*: Device selection options for switching.
- **multi_anchor_cylce**: Number of epochs in between switching (only relevant for *cycle* and *randomized*).
- **multi_anchor_list**: List of (groups of) devices to use for multi anchor training.

# Adaptation Layers
Use adaptable layers during training. 
- **adaptation_layer**: Enable adaptation layers.
- **adaptation_architecture**: Define the feature extractor architecture (a = adaptation layer, c = convolution layer).
- **adaptation_groups**: Define devices that contribute to sub layer training.
- **adaptation_classifier**: Enable adaptation layers in the classification header.
- **reg_loss**: Enable regulation loss between sub layers during **feature extractor training**.
- **adapt_mode_training**: Switch between regular and adaptation mode within the adaptation layers during **feature extractor training**

# Generalizability option
Disregard data from evaluation device during **feature extractor training**.
- **generlizability**: Enable generalizability training.

For further details regarding any of the alterations, refer to the **Implementation** chapter of the [thesis](Master_Thesis.pdf#page=18).
