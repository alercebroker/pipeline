import tensorflow as tf

def balanced_xentropy(labels, predicted_logits):
    predictions = tf.nn.softmax(predicted_logits)
    num_classes = tf.shape(predictions)[1]

    def compute_class_loss(class_index):
        class_mask = tf.equal(labels, class_index)
        class_mask = tf.cast(class_mask, tf.float32)

        class_probs = predictions[:, class_index]  # get probs for this class

        # Only keep class_probs where labels == class_index
        selected_probs = tf.boolean_mask(class_probs, class_mask > 0)

        # Avoid log(0)
        selected_probs = tf.clip_by_value(selected_probs, 1e-15, 1.0 - 1e-15)

        return -tf.reduce_mean(tf.math.log(selected_probs))

    class_losses = tf.map_fn(compute_class_loss, tf.cast(tf.range(num_classes), labels.dtype), dtype=tf.float32)
    return tf.reduce_mean(class_losses)

#def balanced_xentropy(labels, predicted_logits):
#    predictions = tf.nn.softmax(predicted_logits).numpy()
#    scores = []
#    for class_index in range(predictions.shape[1]):
#        objects_from_class = labels == class_index
#        class_probs = predictions[objects_from_class, class_index]
#        class_probs = np.clip(class_probs, 10**-15, 1 - 10**-15)
#        class_score = - np.mean(np.log(class_probs))
#        scores.append(class_score)
#    return np.array(scores).mean()