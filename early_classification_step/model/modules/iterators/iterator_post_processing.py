import tensorflow as tf

# """
# given an image batch tensor as [batch_size,height,width,channels],
# generate 4 rotated versions in [0,90,180,270] degrees and concatenate them in batch dimension
# @param img_batch input tensor thath contains images
# @return a [4*batch_size,height,width,channels] version of img_batch
# """
#
#
# def augment_with_rotations(img_batch):
#   # perform rotations
#   images90 = tf.map_fn(lambda x: tf.image.rot90(x, k=1), img_batch)
#   images180 = tf.map_fn(lambda x: tf.image.rot90(x, k=2), img_batch)
#   images270 = tf.map_fn(lambda x: tf.image.rot90(x, k=3), img_batch)
#
#   # concatenate along first tensor dimension (batch dimension)
#   return tf.concat([img_batch,
#                     images90,
#                     images180,
#                     images270], 0)


"""
TODO: Make a model layer class like LRP, but when back ropagated output means (This would not enable rotations visualizations)
TODO: make callable object to avoid data replication between with labels and normal
"""

"""
Same functionality as method  augment_with_rotations, but recives labels for iterator pipeline
"""


def augment_with_rotations_features(img_batch, features, labels):
    # perform rotations
    images90 = tf.image.rot90(img_batch, k=1)
    images180 = tf.image.rot90(img_batch, k=2)
    images270 = tf.image.rot90(img_batch, k=3)

    # concatenate along first tensor dimension (batch dimension)
    return tf.concat([img_batch, images90, images180, images270], 0), features, labels


def augment_with_rotations_single(img_batch):
    # perform rotations
    images90 = tf.image.rot90(img_batch, k=1)
    images180 = tf.image.rot90(img_batch, k=2)
    images270 = tf.image.rot90(img_batch, k=3)

    # concatenate along first tensor dimension (batch dimension)
    return tf.concat([img_batch, images90, images180, images270], 0)


def augment_with_rotations(img_batch, labels):
    # perform rotations
    images90 = tf.image.rot90(img_batch, k=1)
    images180 = tf.image.rot90(img_batch, k=2)
    images270 = tf.image.rot90(img_batch, k=3)

    # concatenate along first tensor dimension (batch dimension)
    return tf.concat([img_batch, images90, images180, images270], 0), labels


def augment_with_rotations_hm(img_batch, hm, labels):
    # perform rotations
    images90 = tf.image.rot90(img_batch, k=1)
    images180 = tf.image.rot90(img_batch, k=2)
    images270 = tf.image.rot90(img_batch, k=3)

    # concatenate along first tensor dimension (batch dimension)
    return tf.concat([img_batch, images90, images180, images270], 0), hm, labels
