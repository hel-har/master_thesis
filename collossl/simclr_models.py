import tensorflow as tf
tf.random.set_seed(42)

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

@tf.keras.utils.register_keras_serializable()
class SelectLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelectLayer, self).__init__(**kwargs)

    def call(self, inputs):
        conv_outputs_list, selection_index = inputs
        # conv_outputs_list: list of tensors [batch, time, channels]
        # selection_index: scalar tensor or shape [batch] int32/64
        
        stacked = tf.stack(conv_outputs_list, axis=0)  # [num_paths, batch, time, channels]
        
        selection_index = tf.cast(selection_index, tf.int32)
        
        # If selection_index is a scalar:
        if selection_index.shape.rank == 0:
            selected_output = stacked[selection_index]
        else:
            # If per-sample selection is desired (less common), use gather.
            # Assumes selection_index shape [batch], stacked shape [num_paths, batch, ...]
            batch_indices = tf.range(tf.shape(selection_index)[0], dtype=tf.int32)
            indices = tf.stack([selection_index, batch_indices], axis=1)  # [batch, 2]
            selected_output = tf.gather_nd(stacked, indices)
        
        return selected_output

    def get_config(self):
        config = super(SelectLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class AdaptationLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 num_layers=3, 
                 sub_layer_type=tf.keras.layers.Dense, 
                 sub_layer_kwargs=None, 
                 adapt_mode=False, 
                 **kwargs):
        """
        Adaptation layer with multiple sublayers (Dense or Conv) and optional routing.

        Args:
            num_layers: Number of sublayers.
            sub_layer_type: Layer class, e.g., Dense or Conv1D.
            sub_layer_kwargs: Dictionary of kwargs for sublayers.
            adapt_mode: If True, route inputs through chosen branch, else average.
        """
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.adapt_mode = adapt_mode
        self.sub_layer_type = sub_layer_type
        self.sub_layer_kwargs = sub_layer_kwargs or {}

        # Create sublayers
        self.sub_layers = [
            sub_layer_type(**self.sub_layer_kwargs, name=f"adapt_sub_layer_{i}")
            for i in range(self.num_layers)
        ]

    def build(self, input_shape):
        # input_shape can be TensorShape or tuple/list
        if isinstance(input_shape, (list, tuple)):
            data_shape = tf.TensorShape(input_shape[0])
        else:
            data_shape = tf.TensorShape(input_shape)

        for layer in self.sub_layers:
            if not layer.built:
                layer.build(data_shape)
                    
        super().build(input_shape)

    def call(self, inputs):
        # Support (data, chosen_idx) or just data
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            data, chosen_idx = inputs
            chosen_idx = tf.convert_to_tensor(chosen_idx, dtype=tf.int32)
        else:
            data = inputs
            chosen_idx = None

        # Compute all sublayer outputs
        branch_outputs = [layer(tf.identity(data)) for layer in self.sub_layers]
        stacked = tf.stack(branch_outputs, axis=0)  # shape: (num_layers, batch, ...)

        @tf.custom_gradient
        def choose_and_route(stacked_out, idx, adapt_mode_bool):
            def forward():
                def adapt_true():
                    if idx is None:
                        return tf.reduce_mean(stacked_out, axis=0)
                    elif idx.shape.rank == 0:
                        return stacked_out[idx]
                    else:
                        batch_size = tf.shape(idx)[0]
                        batch_range = tf.range(batch_size, dtype=tf.int32)
                        indices = tf.stack([idx, batch_range], axis=1)
                        return tf.gather_nd(stacked_out, indices)
                def adapt_false():
                    return tf.reduce_mean(stacked_out, axis=0)
                return tf.cond(adapt_mode_bool, adapt_true, adapt_false)

            def grad(dy):
                num_branches = tf.shape(stacked_out)[0]

                def grad_true():
                    if idx is None or idx.shape.rank == 0:
                        mask = tf.one_hot(idx if idx is not None else 0, depth=num_branches, dtype=dy.dtype)
                        mask = tf.reshape(mask, (num_branches,) + (1,) * (len(dy.shape)))
                        return dy * mask
                    else:
                        batch_size = tf.shape(idx)[0]
                        batch_range = tf.range(batch_size, dtype=tf.int32)
                        indices = tf.stack([idx, batch_range], axis=1)
                        mask = tf.tensor_scatter_nd_update(tf.zeros_like(stacked_out), indices, dy)
                        return mask

                def grad_false():
                    return tf.tile(tf.expand_dims(dy, 0), [num_branches] + [1] * len(dy.shape))

                return tf.cond(adapt_mode_bool, grad_true, grad_false), None, None

            adapt_mode_const = tf.constant(self.adapt_mode, dtype=tf.bool)
            return forward(), grad

        return choose_and_route(stacked, chosen_idx, tf.constant(self.adapt_mode, dtype=tf.bool))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            return self.sub_layers[0].compute_output_shape(input_shape[0])
        return self.sub_layers[0].compute_output_shape(input_shape)


    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_layers": self.num_layers,
            "adapt_mode": self.adapt_mode,
            "sub_layer_type": tf.keras.utils.serialize_keras_object(self.sub_layers[0]),
            "sub_layer_kwargs": self.sub_layer_kwargs, 
            "sub_layers_config": [layer.get_config() for layer in self.sub_layers],
        })
        return cfg

    @classmethod
    def from_config(cls, config):
        num_layers = config.pop("num_layers")
        adapt_mode = config.pop("adapt_mode")
        sub_layer_template = tf.keras.utils.deserialize_keras_object(config.pop("sub_layer_type"))
        sub_layers_config = config.pop("sub_layers_config")
        sub_layer_kwargs = config.pop("sub_layer_kwargs")

        sub_layer_type = sub_layer_template.__class__
        sub_layers = [sub_layer_type.from_config(c) for c in sub_layers_config]

        layer = cls(num_layers=num_layers,
                    sub_layer_type=sub_layer_type,
                    sub_layer_kwargs=sub_layer_kwargs,
                    adapt_mode=adapt_mode,
                    **config)
        layer.sub_layers = sub_layers
        return layer


def create_base_model(input_shape, num_adaptation_layers=0, adaptation_architecture="c-c-c", model_name="base_model"):
    """
    Create the base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.

    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
        num_adaptation_layers
            the number of convolutional layers that should be used to adapt to specific training devices (if 0, no adjustments will be done)
        adapation_architecture
            which convolutional layer to replace with the adaptation layer
    
    Returns:
        model (tf.keras.Model), last_freeze_layer = 12
    """

    inputs = tf.keras.Input(shape=input_shape, name='input')

    x = inputs
    layer_list = list(adaptation_architecture.split('-'))

    assert len(layer_list) == 3, "Only 3 layers are permitted within the architecture!"

    if 'a' in layer_list:
        assert num_adaptation_layers > 0, "Adaptation layers can only be used with args.adaptation_layer enabled. Use the --adaptation_layer option"
        selection_input = tf.keras.Input(shape=(), dtype=tf.int32, name='layer_selection')

    for idx, layer in enumerate(layer_list):
        filters = 32*(idx+1)
        kernel_size = 24-(8*idx)
        if layer == 'c':
            x = tf.keras.layers.Conv1D(
                    filters, kernel_size,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2=1e-4)
                )(x)
        elif layer == 'a':
            #x = AdaptationLayer(num_adaptation_layers, tf.keras.layers.Conv1D, [filters, kernel_size, 1, 'same', None, 1, 1, 'relu'], adapt_mode=True)([x, selection_input])
            x = AdaptationLayer(num_adaptation_layers, tf.keras.layers.Conv1D, sub_layer_kwargs={"filters": filters, "kernel_size": kernel_size, "padding": 'same', "activation": 'relu'}, adapt_mode=True)([x, selection_input])
        x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(x)

    if 'a' in layer_list:
        model = tf.keras.Model([inputs, selection_input], outputs=x, name=model_name)
    else:
        model = tf.keras.Model(inputs, outputs=x, name=model_name)

    last_freeze_layer = 4
    if num_adaptation_layers > 0:
        last_freeze_layer += 1
    return model, last_freeze_layer


def attach_simclr_head(base_model, hidden_1=256, hidden_2=128, hidden_3=50):
    """
    Attach a 3-layer fully-connected encoding head

    Architecture:
        base_model
        -> Dense: hidden_1 units
        -> ReLU
        -> Dense: hidden_2 units
        -> ReLU
        -> Dense: hidden_3 units
    """

    input = base_model.input
    x = base_model.output

    projection_1 = tf.keras.layers.Dense(hidden_1)(x)
    projection_1 = tf.keras.layers.Activation("relu")(projection_1)
    projection_2 = tf.keras.layers.Dense(hidden_2)(projection_1)
    projection_2 = tf.keras.layers.Activation("relu")(projection_2)
    projection_3 = tf.keras.layers.Dense(hidden_3)(projection_2)

    simclr_model = tf.keras.Model(input, projection_3, name= base_model.name + "_simclr")

    return simclr_model


def create_linear_model_from_base_model(base_model, output_shape, intermediate_layer=-1):

    """
    Create a linear classification model from the base model, using activitations from an intermediate layer

    Architecture:
        base_model-intermediate_layer
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: SGD
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = base_model.inputs
    x = base_model.layers[intermediate_layer].output
    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=base_model.name + "linear")

    if intermediate_layer == -1:
        for i in range(len(base_model.layers)):
            model.layers[i].trainable = False
    else:
        for layer in model.layers[:intermediate_layer+1]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def create_full_classification_model_from_base_model(base_model, output_shape, optimizer, model_name="TPN", intermediate_layer=-1, last_freeze_layer=-1):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing

    Architecture:
        base_model-intermediate_layer
        -> Dense: 1024 units
        -> ReLU
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        model_name
            name of the output model

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    # inputs = base_model.inputs
    intermediate_x = base_model.layers[intermediate_layer].output

    x = tf.keras.layers.Dense(1024, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    for layer in model.layers:
        layer.trainable = False
    
    for layer in model.layers[last_freeze_layer+1:]:
        layer.trainable = True

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),#from_logits=False, reduction=tf.keras.losses.Reduction.NONE),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def create_multi_classification_model_from_base_model(base_model, num_sub_layers, output_shape, optimizer, model_name="TPN", intermediate_layer=-1, last_freeze_layer=-1):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing. Uses Adaptation Layers for the Dense layers of base model uses Adaptation Layers

    Architecture:
        base_model-intermediate_layer
        -> Dense/AdaptationLayer (Dense): 1024 units
        -> ReLU
        -> Dense/AdaptationLayer (Dense): output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy

    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories

        model_name
            name of the output model

        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)

    """

    adaptation_layer = False

    for layer in base_model.layers:
        if isinstance(layer, AdaptationLayer):
            adaptation_layer = True
            num_layers = len(layer.sub_layers)
            break
    if not adaptation_layer:
        selection_input = tf.keras.Input(shape=(), dtype=tf.int32, name='layer_selection')
        inputs = [base_model.inputs[0], selection_input]
    
    else:
        inputs = inputs = base_model.inputs
    intermediate_x = base_model.layers[intermediate_layer].output

    x = AdaptationLayer(num_layers=num_sub_layers, sub_layer_type=tf.keras.layers.Dense, sub_layer_kwargs={"units": 1024, "activation": 'relu'}, adapt_mode=True)([intermediate_x, inputs[1]])
    x = AdaptationLayer(num_layers=num_sub_layers, sub_layer_type=tf.keras.layers.Dense, sub_layer_kwargs={"units": output_shape}, adapt_mode=True)([x, inputs[1]])

    outputs = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    for layer in model.layers:
        layer.trainable = False
    
    for layer in model.layers[last_freeze_layer+1:]:
        layer.trainable = True

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),#from_logits=False, reduction=tf.keras.losses.Reduction.NONE),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    model.summary()

    return model


def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    """
    Create an intermediate model from base mode, which outputs embeddings of the intermediate layer

    Parameters:
        base_model
            the base model from which the intermediate model is built
        
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted

    Returns:
        model (tf.keras.Model)
    """

    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

