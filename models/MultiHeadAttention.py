import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout,Dense,Layer
# import tf.contrib.layers.layer_norm
# from keras.layers.normalization.batch_normalization_v1 import BatchNormalization,LayerNormalization
from keras_layer_normalization import LayerNormalization

class MultiHeadSelfAttention_Model(Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention_Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim,use_bias=False)
        self.key_dense = Dense(embed_dim,use_bias=False)
        self.value_dense = Dense(embed_dim,use_bias=False)
        self.combine_heads = Dense(embed_dim,use_bias=False)
        


    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

        query = self.separate_heads(
            query, batch_size
        )  
        key = self.separate_heads(
            key, batch_size
        )  
        value = self.separate_heads(
            value, batch_size
        )  
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  
        output = self.combine_heads(
            concat_attention
        )  
        return output