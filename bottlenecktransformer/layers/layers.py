from einops import rearrange
import tensorflow as tf
from bottlenecktransformer.utils.version_resolver import get_policy
from tensorflow.keras import layers as L


def rel_to_abs(x, heads_by_width, length):
    h = heads_by_width
    l = length

    x = tf.pad(x, paddings=[[0, 0], [0, 0], [0, 0], [0, 1]])

    flat_x = rearrange(x, "b h l c -> b h (l c)")

    flat_x_padded = tf.pad(flat_x, paddings=[[0, 0], [0, 0], [0, l - 1]])

    final_x = tf.reshape(flat_x_padded, (-1, h, l + 1, 2 * l - 1))
    final_x = final_x[:, :, :l, (l - 1) :]

    return final_x


def expand_dim(t, dim, k):
    t = tf.expand_dims(t, axis=dim)
    t = tf.repeat(t, k, axis=dim)
    return t


def relative_logits_1d(q, rel_k, heads, h, w):
    logits = tf.einsum("b h x y d, r d -> b h x y r", q, rel_k)
    logits = rearrange(logits, "b h x y r -> b (h x) y r")

    logits = rel_to_abs(logits, heads * w, w)
    logits = tf.reshape(logits, (-1, heads, h, w, w))
    logits = expand_dim(logits, dim=3, k=h)

    return logits


class RelPosEmb(tf.keras.layers.Layer):
    def __init__(self, dimensions_per_head, real_input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensions_per_head = dimensions_per_head
        self.real_input_shape = real_input_shape

    def build(self, input_shape):
        b, h, w, c = self.real_input_shape
        self.heads = input_shape[1]
        self.scale = self.dimensions_per_head ** -0.5

        self.height = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(h * 2 - 1, self.dimensions_per_head)
            ),
            dtype="float32",
            name="height",
        )
        self.width = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(w * 2 - 1, self.dimensions_per_head)
            ),
            dtype="float32",
            name="width",
        )

    def call(self, q):
        b, h, w, c = self.real_input_shape

        q = rearrange(q, "b h (x y) d -> b h x y d", x=h, y=w)
        rel_logits_w = relative_logits_1d(q, self.width, self.heads, h, w)
        rel_logits_w = rearrange(rel_logits_w, "b h x i y j-> b h (x y) (i j)")

        q = rearrange(q, "b h x y d -> b h y x d")
        rel_logits_h = relative_logits_1d(q, self.height, self.heads, h, w)
        rel_logits_h = rearrange(rel_logits_h, "b h x i y j -> b h (y x) (j i)")
        return rel_logits_w + rel_logits_h

    def get_config(self):
        config = super().get_config()
        config["dimensions_per_head"] = self.dimensions_per_head
        config["real_input_shape"] = self.real_input_shape
        return config


class AbsEmb(tf.keras.layers.Layer):
    def __init__(self, dimensions_per_head, real_input_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensions_per_head = dimensions_per_head
        self.real_input_shape = real_input_shape

    def build(self, input_shape):
        b, h, w, c = self.real_input_shape

        self.scale = self.dimensions_per_head ** -0.5

        self.height = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(h, self.dimensions_per_head)
            ),
            dtype="float32",
            name="height",
        )
        self.width = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(w, self.dimensions_per_head)
            ),
            dtype="float32",
            name="width",
        )

    def call(self, q):
        emb = rearrange(self.height, "h d -> h () d") + rearrange(
            self.width, "w d -> () w d"
        )
        emb = rearrange(emb, " h w d -> (h w) d")
        emb = tf.cast(emb, get_policy().compute_dtype)

        logits = tf.einsum("b h i d, j d -> b h i j", q, emb)
        return logits

    def get_config(self):
        config = super().get_config()
        config["dimensions_per_head"] = self.dimensions_per_head
        config["real_input_shape"] = self.real_input_shape
        return config


class Attention(tf.keras.layers.Layer):
    def __init__(
        self, dimensions_per_head=128, heads=4, rel_pos_emb=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dimensions_per_head = dimensions_per_head
        self.heads = 4
        self.scale = dimensions_per_head ** -0.5
        self.pos_emb_method = RelPosEmb if rel_pos_emb else AbsEmb
        self.rel_pos_emb = rel_pos_emb

    def build(self, input_shape):
        b, h, w, c = input_shape

        self.to_qkv = tf.keras.layers.Conv2D(
            filters=self.dimensions_per_head * self.heads * 3,
            kernel_size=1,
            use_bias=False,
        )

        self.pos_emb = self.pos_emb_method(self.dimensions_per_head, input_shape)

        self.h = h
        self.w = w

    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b h w c -> b c h w")

        q, k, v = tf.split(qkv, 3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=self.heads),
            (q, k, v),
        )

        q *= self.scale

        sim = tf.einsum("b h i d, b h j d -> b h i j", q, k)
        sim += self.pos_emb(q)

        attn = tf.nn.softmax(sim, axis=-1)

        out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=self.h, y=self.w)
        out = rearrange(out, "b c h w -> b h w c")
        return out

    def get_config(self):
        config = super().get_config()
        config["dimensions_per_head"] = self.dimensions_per_head
        config["heads"] = self.heads
        config["rel_pos_emb"] = self.rel_pos_emb
        return config


class BottleBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        output_channels,
        dimensions_per_head=128,
        heads=4,
        projection_factor=4,
        downsample=False,
        pool_downsample=False,
        activation="relu",
        rel_pos_emb=False,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.dimensions_per_head = dimensions_per_head
        self.heads = 4
        self.scale = dimensions_per_head ** -0.5
        self.projection_factor = projection_factor
        self.activation = activation
        self.output_channels = output_channels
        self.downsample = downsample
        self.pool_downsample = pool_downsample
        self.rel_pos_emb = rel_pos_emb

    def build(self, input_shape):
        b, h, w, c = input_shape

        att_c_in = self.output_channels // self.projection_factor

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(att_c_in, 1),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.activation),
                Attention(self.dimensions_per_head, self.heads, self.rel_pos_emb),
                tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
                if self.downsample
                else L.Activation(tf.identity),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.Conv2D(self.output_channels, 1),
                tf.keras.layers.BatchNormalization(),
            ]
        )

        if c != self.output_channels or self.downsample:
            if not self.pool_downsample:

                kernel_size, stride = (3, 2) if self.downsample else (1, 1)

                self.shortcut = tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(
                            self.output_channels,
                            kernel_size,
                            strides=stride,
                            padding="same",
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation(self.activation),
                    ]
                )
            else:
                self.shortcut = tf.keras.Sequential(
                    [
                        tf.keras.layers.AveragePooling2D((2, 2)),
                        tf.keras.layers.Conv2D(
                            self.output_channels, kernel_size=1, padding="same"
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.Activation(self.activation),
                    ]
                )

        else:
            self.shortcut = L.Activation(tf.identity)

    def call(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return L.Activation(self.activation)(x)

    def get_config(self):
        config = super().get_config()
        config["dimensions_per_head"] = self.dimensions_per_head
        config["heads"] = self.heads
        config["projection_factor"] = self.projection_factor
        config["activation"] = self.activation
        config["output_channels"] = self.output_channels
        config["downsample"] = self.downsample
        config["pool_downsample"] = self.pool_downsample
        config["rel_pos_emb"] = self.rel_pos_emb
        return config
