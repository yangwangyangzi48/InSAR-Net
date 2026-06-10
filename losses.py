import tensorflow as tf


def phase_from_vector(y):
    return tf.atan2(y[..., 1], y[..., 0])


def wrap(x):
    return tf.atan2(tf.sin(x), tf.cos(x))


def phase_dx(phi):
    return wrap(phi[:, :, 1:] - phi[:, :, :-1])


def phase_dy(phi):
    return wrap(phi[:, 1:, :] - phi[:, :-1, :])


def complex_plane_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))


def angular_loss(y_true, y_pred):
    pt = phase_from_vector(y_true)
    pp = phase_from_vector(y_pred)
    return tf.reduce_mean(tf.square(tf.sin(wrap(pt - pp))))


def gradient_loss(y_true, y_pred):
    pt = phase_from_vector(y_true)
    pp = phase_from_vector(y_pred)
    loss_x = tf.reduce_mean(tf.abs(phase_dx(pt) - phase_dx(pp)))
    loss_y = tf.reduce_mean(tf.abs(phase_dy(pt) - phase_dy(pp)))
    return loss_x + loss_y


def gradient_consistency_loss(y_true, y_pred):
    pt = phase_from_vector(y_true)
    pp = phase_from_vector(y_pred)
    gtx = phase_dx(pt)
    gty = phase_dy(pt)
    gpx = phase_dx(pp)
    gpy = phase_dy(pp)
    gtx = gtx[:, :-1, :]
    gpx = gpx[:, :-1, :]
    gty = gty[:, :, :-1]
    gpy = gpy[:, :, :-1]
    dot = gtx * gpx + gty * gpy
    nt = tf.sqrt(tf.square(gtx) + tf.square(gty) + 1e-8)
    npred = tf.sqrt(tf.square(gpx) + tf.square(gpy) + 1e-8)
    cos_sim = dot / (nt * npred + 1e-8)
    return tf.reduce_mean(1.0 - cos_sim)


def phase_consistency_loss(lambda_cmp=0.4, lambda_ang=0.1, lambda_grad=0.3, lambda_cons=0.2):
    def loss(y_true, y_pred):
        return lambda_cmp * complex_plane_loss(y_true, y_pred) + lambda_ang * angular_loss(y_true, y_pred) + lambda_grad * gradient_loss(y_true, y_pred) + lambda_cons * gradient_consistency_loss(y_true, y_pred)
    return loss
