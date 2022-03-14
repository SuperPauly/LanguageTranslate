#wget https://data.statmt.org/rsennrich/wmt16_backtranslations/en-ru/news.bt.en-ru.en.gz && wget https://data.statmt.org/rsennrich/wmt16_backtranslations/en-ru/news.bt.en-ru.ru.gz && gzip -vdf news.bt.en-ru.en.gz && gzip -vdf news.bt.en-ru.ru.gz
import pathlib
import random
import string
import argparse
from time import time
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import horovod.tensorflow.keras as hvd


# Horovod: initialize Horovod.
hvd.init()


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

bs = 64 * hvd.size()

russian = []
english = []

with open("news.bt.en-ru.en", "r", encoding='utf8', errors="ignore") as en:
    english.append(en.read().split("\n"))
english = english[0]
with open("news.bt.en-ru.ru", "r", encoding='utf8', errors="ignore") as ru:
    russian.append(ru.read().split("\n"))
russian = russian[0]

dataset = zip(russian, english)
dataset = list(dataset)[0:1500000]


random.shuffle(dataset)
num_val_samples = int(0.15 * len(dataset))
num_train_samples = len(dataset) - 2 * num_val_samples
train_pairs = dataset[:num_train_samples]
print(len(train_pairs))
val_pairs = dataset[num_train_samples : num_train_samples + num_val_samples]
test_pairs = dataset[num_train_samples + num_val_samples :]

print(f"{len(dataset)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

vocab_size = 15000
sequence_length = 50
batch_size = bs
eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
)
rus_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length + 1,
)

train_eng_texts = [pair[1] for pair in train_pairs]
train_rus_texts = [pair[0] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
rus_vectorization.adapt(train_rus_texts)

def format_dataset(eng, rus):
    eng = eng_vectorization(eng)
    rus = rus_vectorization(rus)
    return ({"encoder_inputs": eng, "decoder_inputs": rus[:, :-1],}, rus[:, 1:])


def make_dataset(pairs):
    rus_texts, eng_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    rus_texts = list(rus_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, rus_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(500000, reshuffle_each_iteration=True).prefetch(50000).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print("\n\n")
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")
    print("\n\n")

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
            'attention': self.attention,
            'dense_proj': self.dense_proj,
            'layernorm_1': self.layernorm_1,
            'layernorm_2': self.layernorm_2,
            'supports_masking': self.supports_masking,
        })
        return config

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_embeddings': self.token_embeddings,
            'position_embeddings': self.position_embeddings,
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'attention_1': self.attention_1,
            'attention_2': self.attention_2,
            'dense_proj': self.dense_proj,
            'layernorm_1': self.layernorm_1,
            'layernorm_2': self.layernorm_2,
            'layernorm_3': self.layernorm_3,
            'supports_masking': self.supports_masking
        })
        return config

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

embed_dim = 256
latent_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")

optimizer = tf.keras.optimizers.RMSprop()
optimizer = hvd.DistributedOptimizer(optimizer)


cb = []
cb.append(hvd.callbacks.LearningRateScheduleCallback(initial_lr=0.0002 * hvd.size(), multiplier=0.0003 * hvd.size(), start_epoch=15, staircase=True, steps_per_epoch=4))
cb.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
cb.append(hvd.callbacks.MetricAverageCallback())
if hvd.rank() == 0:
    cb.append(tf.keras.callbacks.ModelCheckpoint('./ck.hdf5', save_freq=1, save_weights_only=False))


    
epochs = 25  # This should be at least 30 for convergence

transformer.summary()
transformer.compile(optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
transformer.fit(train_ds, epochs=epochs, shuffle=True, validation_data=val_ds, validation_steps=len(val_ds), callbacks=cb, verbose=1, workers=1)
