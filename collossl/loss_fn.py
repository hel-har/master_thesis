import tensorflow as tf
from simclr_models import AdaptationLayer
tf.random.set_seed(42)

# @tf.function
def weighted_group_contrastive_loss_with_temp(anchor_embedding, positive_embeddings, positive_weights, negative_embeddings, negative_weights, temperature=2):
    """
    the loss function for group-supervised training 
    it has extra tempearture argument 
    it accepts three arguments: anchor_embedding, positive_embeddings, negative_embeddings
    anchor_embedding has shape (?, e)
    positive_embeddings has shape (p, ?, e)
    negative_embeddings has shape (n, ?, e)
    ? refers to the batch size (variable), e is the embedding size of the model, p is the number of positive devices, n is the number of negative devices
    """
    anchor = tf.convert_to_tensor(anchor_embedding)
    pos_embs = tf.convert_to_tensor(positive_embeddings, anchor.dtype)
    neg_embs = tf.convert_to_tensor(negative_embeddings, anchor.dtype)
    sim = tf.keras.losses.CosineSimilarity(
        axis=-1, reduction=tf.keras.losses.Reduction.NONE)

    pos_sim = sim(tf.broadcast_to(anchor, pos_embs.shape), pos_embs)/temperature  # (p,?)
    neg_sim = sim(tf.broadcast_to(anchor, neg_embs.shape), neg_embs)/temperature  # (n,?)
    numerator = tf.math.log(tf.reduce_sum(tf.exp(pos_sim) * tf.cast(positive_weights, pos_sim.dtype), axis=0))  # (?)
    denominator = tf.math.log(tf.reduce_sum(tf.exp(pos_sim) * tf.cast(positive_weights, pos_sim.dtype), axis=0) +
                         tf.reduce_sum(tf.exp(neg_sim) * tf.cast(negative_weights, neg_sim.dtype), axis=0)) # (?)
    
    return tf.math.reduce_mean(denominator - numerator)

def weighted_group_contrastive_loss_with_temp_and_reg(
    anchor_embedding,
    positive_embeddings,
    positive_weights,
    negative_embeddings,
    negative_weights,
    input_data,
    anchor_device,
    model,
    temperature=2.0,
    lambda_reg=1e-4
):
    """
    Weighted group contrastive loss + activation regularization
    using mean output of all sub-layers as the reference.

    Args:
        anchor_embedding: (?, e) anchor representation
        positive_embeddings: (p, ?, e) positive device embeddings
        positive_weights: (p, ?) weights for positives
        negative_embeddings: (n, ?, e) negative device embeddings
        negative_weights: (n, ?) weights for negatives
        anchor_input: (?, t, c) raw batch input for anchor device
        anchor_device: int, which device is anchor
        sub_layers: dict {device_id: tf.keras.layers.Layer}
        temperature: float, softmax temperature
        lambda_reg: float, regularization strength
    Returns:
        scalar total loss
    """
    anchor = tf.convert_to_tensor(anchor_embedding)
    pos_embs = tf.convert_to_tensor(positive_embeddings, anchor.dtype)
    neg_embs = tf.convert_to_tensor(negative_embeddings, anchor.dtype)
    sim = tf.keras.losses.CosineSimilarity(
        axis=-1, reduction=tf.keras.losses.Reduction.NONE)

    pos_sim = sim(tf.broadcast_to(anchor, pos_embs.shape), pos_embs)/temperature  # (p,?)
    neg_sim = sim(tf.broadcast_to(anchor, neg_embs.shape), neg_embs)/temperature  # (n,?)
    numerator = tf.math.log(tf.reduce_sum(tf.exp(pos_sim) * tf.cast(positive_weights, pos_sim.dtype), axis=0))  # (?)
    denominator = tf.math.log(tf.reduce_sum(tf.exp(pos_sim) * tf.cast(positive_weights, pos_sim.dtype), axis=0) +
                         tf.reduce_sum(tf.exp(neg_sim) * tf.cast(negative_weights, neg_sim.dtype), axis=0)) # (?)

    contrastive_loss = tf.reduce_mean(denominator - numerator)
    
    kl = tf.keras.losses.KLDivergence()

    # --- Multi-layer activation regularization ---
    reg_loss = 0.0
    current_input = input_data  # input to first adaptation layer
    for layer in model.layers:
        if isinstance(layer, AdaptationLayer):

            # Forward pass through all sub-layers for this layer
            outputs = [layer.sub_layers[d](current_input) for d in range(len(layer.sub_layers))]
            
            # Convert to probability distributions
            probs = [tf.nn.softmax(o, axis=-1) for o in outputs]
            mean_probs = tf.add_n(probs) / len(probs)

            # Anchor device distribution
            anchor_probs = tf.nn.softmax(layer.sub_layers[anchor_device](current_input), axis=-1)

            # KL(anchor || mean)

            kl_div = tf.reduce_mean(
                kl.call(anchor_probs, mean_probs)
            )

            reg_loss += lambda_reg * kl_div

            # Pass anchor output forward
            current_input = outputs[anchor_device]
        elif isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Dropout):
            current_input = layer(current_input)

    total_loss = contrastive_loss + reg_loss
    return total_loss