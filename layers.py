import tensorflow as tf


class FSC(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.phase_conv = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")
        self.amp_conv = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")
        self.grad_conv = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")
        self.compress = tf.keras.layers.Conv2D(filters, 1, padding="same", activation="relu")

    def call(self, inputs):
        z1r = inputs[..., 0]
        z1i = inputs[..., 1]
        z2r = inputs[..., 2]
        z2i = inputs[..., 3]
        real = z1r * z2r + z1i * z2i
        imag = z1i * z2r - z1r * z2i
        amp = tf.sqrt(tf.maximum(real * real + imag * imag, 1e-8))
        cos_phi = real / amp
        sin_phi = imag / amp
        phi = tf.atan2(sin_phi, cos_phi)
        dx = tf.pad(phi[:, :, 1:] - phi[:, :, :-1], [[0, 0], [0, 0], [0, 1]])
        dy = tf.pad(phi[:, 1:, :] - phi[:, :-1, :], [[0, 0], [0, 1], [0, 0]])
        dx = tf.atan2(tf.sin(dx), tf.cos(dx))
        dy = tf.atan2(tf.sin(dy), tf.cos(dy))
        phase_features = self.phase_conv(tf.stack([cos_phi, sin_phi], axis=-1))
        amp_features = self.amp_conv(tf.expand_dims(tf.math.log1p(amp), axis=-1))
        grad_features = self.grad_conv(tf.stack([dx, dy], axis=-1))
        return self.compress(tf.concat([phase_features, amp_features, grad_features], axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class RDB(tf.keras.layers.Layer):
    def __init__(self, filters, growth_rate, layers, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.growth_rate = growth_rate
        self.layers_count = layers
        self.convs = [tf.keras.layers.Conv2D(growth_rate, 3, padding="same", activation="relu") for _ in range(layers)]
        self.local_fusion = tf.keras.layers.Conv2D(filters, 1, padding="same")
        self.shortcut = tf.keras.layers.Conv2D(filters, 1, padding="same")

    def call(self, inputs):
        features = [inputs]
        x = inputs
        for conv in self.convs:
            x = conv(tf.concat(features, axis=-1))
            features.append(x)
        fused = self.local_fusion(tf.concat(features, axis=-1))
        shortcut = inputs if inputs.shape[-1] == self.filters else self.shortcut(inputs)
        return tf.nn.relu(fused + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "growth_rate": self.growth_rate, "layers": self.layers_count})
        return config


class AFF(tf.keras.layers.Layer):
    def __init__(self, filters, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.reduction = reduction
        self.context_proj = tf.keras.layers.Conv2D(filters, 1, padding="same")
        self.detail_proj = tf.keras.layers.Conv2D(filters, 1, padding="same")
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.fc1 = tf.keras.layers.Dense(max(filters // reduction, 4), activation="relu")
        self.fc2 = tf.keras.layers.Dense(filters * 2, activation="sigmoid")
        self.refine = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")

    def call(self, inputs):
        context, detail = inputs
        c = self.context_proj(context)
        d = self.detail_proj(detail)
        s = self.gap(c + d)
        w = self.fc2(self.fc1(s))
        wc, wd = tf.split(w, 2, axis=-1)
        wc = tf.reshape(wc, [-1, 1, 1, self.filters])
        wd = tf.reshape(wd, [-1, 1, 1, self.filters])
        return self.refine(wc * c + wd * d)

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "reduction": self.reduction})
        return config
