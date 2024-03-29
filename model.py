import tensorflow as tf

class InputEmbeddings(tf.keras.layers.Layer):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
    
    def call(self, x: tf.Tensor):
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        return x

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.pe = tf.zeros((1, seq_len, d_model))
        positions = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = 1 / (tf.pow(10000, tf.range(0, d_model, 2, dtype=tf.float32) / d_model))

        self.pe[:, :, 0::2] = tf.math.sin(positions * div_term)
        self.pe[:, :, 1::2] = tf.math.cos(positions * div_term)
    
    def call(self, x: tf.Tensor):
        x = x + self.pe[:, :x.shape[1], 1]
        return self.dropout(x)

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()