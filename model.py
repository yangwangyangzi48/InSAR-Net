import tensorflow as tf
from .layers import FSC, RDB, AFF


def conv_block(x, filters, stride=1):
    x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def lateral(x, filters):
    return tf.keras.layers.Conv2D(filters, 1, padding="same")(x)


def refine(x, filters):
    return tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)


def resize_to(x, ref):
    return tf.keras.layers.Lambda(lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method="bilinear"))([x, ref])


def unit_output(x):
    x = tf.keras.layers.Conv2D(2, 1, padding="same")(x)
    return tf.keras.layers.Lambda(lambda t: t / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(t), axis=-1, keepdims=True)), 1e-6), name="phase_vector")(x)


def build_insarnet(image_size=256, base_channels=64, fpn_channels=128, rdb_layers=4, rdb_growth_rate=32):
    inputs = tf.keras.layers.Input((image_size, image_size, 4), name="slc_pair")
    fsc = FSC(base_channels, name="fsc")(inputs)
    detail = conv_block(fsc, base_channels)
    detail = conv_block(detail, fpn_channels)
    x = conv_block(fsc, base_channels, stride=2)
    x = conv_block(x, base_channels, stride=2)
    c2 = RDB(base_channels, rdb_growth_rate, rdb_layers, name="rdb_c2")(x)
    x = conv_block(c2, base_channels * 2, stride=2)
    c3 = RDB(base_channels * 2, rdb_growth_rate, rdb_layers, name="rdb_c3")(x)
    x = conv_block(c3, base_channels * 4, stride=2)
    c4 = RDB(base_channels * 4, rdb_growth_rate, rdb_layers, name="rdb_c4")(x)
    x = conv_block(c4, base_channels * 8, stride=2)
    c5 = RDB(base_channels * 8, rdb_growth_rate, rdb_layers, name="rdb_c5")(x)
    m2 = lateral(c2, fpn_channels)
    m3 = lateral(c3, fpn_channels)
    m4 = lateral(c4, fpn_channels)
    p5 = refine(lateral(c5, fpn_channels), fpn_channels)
    p4 = refine(m4 + resize_to(p5, m4), fpn_channels)
    p3 = refine(m3 + resize_to(p4, m3), fpn_channels)
    p2 = refine(m2 + resize_to(p3, m2), fpn_channels)
    context = resize_to(p2, detail)
    fused = AFF(fpn_channels, name="aff")([context, detail])
    x = conv_block(fused, fpn_channels)
    x = conv_block(x, base_channels)
    outputs = unit_output(x)
    return tf.keras.Model(inputs, outputs, name="InSAR-Net")


def custom_objects():
    return {"FSC": FSC, "RDB": RDB, "AFF": AFF}
