"""Define functions to create the triplet loss with online triplet mining."""
import logging
import time

import numpy as np
import tensorflow as tf

from pyxtools import calc_distance_pairs


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets


def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.to_float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.to_float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss


def _get_pair_sum_loss(xi, margin, squared=False):
    d12 = tf.linalg.norm(xi[0] - xi[1] + 1e-16)
    d13 = tf.linalg.norm(xi[0] - xi[2] + 1e-16)
    d14 = tf.linalg.norm(xi[0] - xi[3] + 1e-16)
    d23 = tf.linalg.norm(xi[1] - xi[2] + 1e-16)
    d24 = tf.linalg.norm(xi[1] - xi[3] + 1e-16)

    if squared:
        d12 = tf.square(d12)
        d13 = tf.square(d13)
        d14 = tf.square(d14)
        d23 = tf.square(d23)
        d24 = tf.square(d24)

    return tf.maximum(d12 - d13 + margin, 0.0) + \
           tf.maximum(d12 - d14 + margin, 0.0) + \
           tf.maximum(d12 - d23 + margin, 0.0) + \
           tf.maximum(d12 - d24 + margin, 0.0)


def test_numpy():
    import numpy as np

    def calc_loss(x) -> float:
        loss_list = []
        for i in range(x.shape[0]):
            d12 = np.linalg.norm(x[i][0] - x[i][1])
            d13 = np.linalg.norm(x[i][0] - x[i][2])
            d14 = np.linalg.norm(x[i][0] - x[i][3])
            d23 = np.linalg.norm(x[i][1] - x[i][2])
            d24 = np.linalg.norm(x[i][1] - x[i][3])
            loss_list.append(np.maximum(d12 - d13 + 0.5, 0.0))
            loss_list.append(np.maximum(d12 - d14 + 0.5, 0.0))
            loss_list.append(np.maximum(d12 - d23 + 0.5, 0.0))
            loss_list.append(np.maximum(d12 - d24 + 0.5, 0.0))

        return float(np.mean(loss_list))

    def calc_loss_tf(x) -> float:
        tx = tf.constant(x)
        loss_list = tf.map_fn(lambda xi: _get_pair_sum_loss(xi, 0.5, squared=False), elems=tx)
        loss = tf.reduce_mean(loss_list) / 4
        with tf.Session() as sess:
            return float(sess.run(loss))

    y = np.random.random((128, 32))
    y = y.reshape((-1, 4, 32))
    print("loss of np is {}, loss of tf is {}".format(calc_loss(y), calc_loss_tf(y)))
    assert (calc_loss(y) - calc_loss_tf(y)) < 1e-5


def batch_hard_triplet_loss_v2(labels, embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.


    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    embedding_pairs = tf.reshape(embeddings, (-1, 4, embeddings.shape[-1]))
    triplet_loss = tf.map_fn(lambda xi: _get_pair_sum_loss(xi, margin, squared=squared), elems=embedding_pairs)
    triplet_loss = tf.divide(tf.reduce_mean(triplet_loss), 4.0)

    return triplet_loss


def get_triplet_pair_np(anchor_feature: np.ndarray, all_feature: np.ndarray, all_label: np.ndarray,
                        margin: float, logger=logging) -> list:
    """
        get triplet pair by hardest mode
    Args:
        logger: Logger,
        anchor_feature: anchor feature, normalized,  sort by class; shape is (m, n), n is feature dimension.
                        Sort like this, A1, A2, B1, B2, B3, C1, C2...
        all_feature: all feature, normalized, sort by class; shape is (m+k, n),  n is feature dimension.
                        all_feature[:m]==anchor_feature. Value like this, A, A, B, B, B, C, C...
        all_label: class of feature; shape is (m+k, ).
        margin: float.
    Returns:
        list:  apn pair list. Like [(0, 2, 15), (3, 4, 15), ...]. 0, 2, 15, is the index in all_feature.
    """
    # todo pairwise_distances 出现nan (float对比大小会出错; np对比则正常);
    # todo normalized; margin
    # distance
    _time_start = time.time()
    pairwise_distances = calc_distance_pairs(anchor_feature, all_feature)  # m x (m+l)
    logger.info("success to calculate distance matrix, time cost {}s!".format(time.time() - _time_start))

    # find apn pair
    result_list = []
    m, n, k = anchor_feature.shape[0], anchor_feature.shape[1], all_feature.shape[0] - anchor_feature.shape[0]
    logger.info("m is {}, n is {}, k is {}".format(m, n, k))

    _time_start = time.time()
    positive_label_record = {}
    _start_index = -1
    _start_label = None
    for a_index in range(m):
        if all_label[a_index] != _start_label:
            _start_label = all_label[a_index]
            _start_index = a_index

        positive_label_record[a_index] = []
        for p_index in range(_start_index, m):
            if all_label[a_index] == all_label[p_index]:
                positive_label_record[a_index].append(p_index)
            else:
                break
        if len(positive_label_record[a_index]) <= 1:
            logger.warning("a_index {} is invalid anchor!".format(a_index))

    logger.info("success to calculate positive_label_record, time cost {}s!".format(time.time() - _time_start))

    # get apn pair
    for a_index, p_index_list in positive_label_record.items():
        # positive
        p_hardest_index = np.argmax(pairwise_distances[a_index][p_index_list[0]: p_index_list[-1] + 1]) + \
                          p_index_list[0]

        # negative
        _distance = pairwise_distances[a_index].copy()
        for _index in p_index_list:
            _distance[_index] = 1e12
        n_hardest_index = np.argmin(_distance)

        if (pairwise_distances[a_index][p_hardest_index] + margin) > pairwise_distances[a_index][n_hardest_index]:
            result_list.append((a_index, p_hardest_index, n_hardest_index))

    return result_list


__all__ = ("batch_all_triplet_loss", "get_triplet_pair_np", "batch_hard_triplet_loss", "batch_hard_triplet_loss_v2")
