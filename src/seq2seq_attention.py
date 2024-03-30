import os

import pandas as pd
import numpy as np

import tensorflow as tf

import jieba

START_TOKEN = "<s>"
END_TOKEN = "</s>"

class DataLoader:
    def __init__(self):
        jieba.enable_parallel(8)

        self.en_context_text_processor = None
        self.zh_context_text_processor = None

        self.train_ds = None
        self.val_ds = None

    def build(self, train_data_path, val_data_path, max_vocab_size=20000):
        train_data = pd.read_csv(train_data_path, header=0)
        val_data = pd.read_csv(val_data_path, header=0)

        BATCH_SIZE = 32

        train_zh_tokenised = [" ".join(jieba.cut(row["zh"])) for _, row in train_data.iterrows()]
        val_zh_tokenised = [" ".join(jieba.cut(row["zh"])) for _, row in val_data.iterrows()]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_data['en'].tolist(), train_zh_tokenised)).shuffle(len(train_data)).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data['en'].tolist(), val_zh_tokenised)).shuffle(len(val_data)).batch(BATCH_SIZE)

        # Build vocab dictionary
        en_context_text_processor = tf.keras.layers.TextVectorization(
            standardize=self._preprocess_en,
            max_tokens=max_vocab_size,
            ragged=True)
        zh_context_text_processor = tf.keras.layers.TextVectorization(
            standardize=self._preprocess_zh,
            max_tokens=max_vocab_size,
            ragged=True)
        
        en_context_text_processor.adapt(train_dataset.map(lambda en, _: en))
        zh_context_text_processor.adapt(train_dataset.map(lambda _, zh: zh))
        self.en_context_text_processor = en_context_text_processor
        self.zh_context_text_processor = zh_context_text_processor

        self.train_ds = train_dataset.map(self._process_text, tf.data.AUTOTUNE)
        self.val_ds = val_dataset.map(self._process_text, tf.data.AUTOTUNE)

    def _preprocess_en(self, en_text):
        en_text = tf.strings.lower(en_text)
        en_text = tf.strings.regex_replace(en_text, '[.?!,]', r' \0 ')
        en_text = tf.strings.join([START_TOKEN, en_text, END_TOKEN], separator=' ')
        en_text = tf.strings.strip(en_text)

        return en_text
    
    def _preprocess_zh(self, zh_text):
        zh_text = tf.strings.join([START_TOKEN, zh_text, END_TOKEN], separator=' ')
        zh_text = tf.strings.strip(zh_text)

        return zh_text
    
    def _process_text(self, en_text, zh_text):
        en_padded = self.en_context_text_processor(en_text).to_tensor()
        zh_id_vector = self.zh_context_text_processor(zh_text)
        target_in = zh_id_vector[:,:-1].to_tensor()
        target_out = zh_id_vector[:,1:].to_tensor()

        return (en_padded, target_in), target_out

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        self.rnn = tf.keras.layers.Bidirectional(merge_mode='sum',
                        layer=tf.keras.layers.GRU(units,
                                    # Return the sequence and state
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform'))

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)

        return x
    
    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]

        context = self.text_processor(texts).to_tensor()
        context = self(context)

        return context

class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x

class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id(START_TOKEN)
        self.end_token = self.word_to_id(END_TOKEN)

        self.units = units

        # 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units)

        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        x = self.embedding(x)

        x, state = self.rnn(x, initial_state=state)

        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights

        logits = self.output_layer(x)

        if return_state:
            return logits, state
        else:
            return logits

    # def get_initial_state(self, context):
    #     batch_size = tf.shape(context)[0]
    #     start_tokens = tf.fill([batch_size, 1], START_TOKEN)
    #     done = tf.zeros([batch_size, 1], dtype=tf.bool)
    #     embedded = self.embedding(start_tokens)
    #     return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result

    # def get_next_token(self, context, next_token, done, state, temperature = 0.0):
    #     logits, state = self(
    #         context, next_token,
    #         state = state,
    #         return_state=True) 

    #     if temperature == 0.0:
    #         next_token = tf.argmax(logits, axis=-1)
    #     else:
    #         logits = logits[:, -1, :]/temperature
    #         next_token = tf.random.categorical(logits, num_samples=1)

    #     # If a sequence produces an `end_token`, set it `done`
    #     done = done | (next_token == self.end_token)
    #     # Once a sequence is done it only produces 0-padding.
    #     next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    #     return next_token, done, state

class Translator(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
    
    def translate(self, input_sequence, max_length=50):
        context = self.encoder.convert_input(input_sequence)
        # context_vectors = self.encoder(context)

        decoder_input = tf.expand_dims([self.decoder.start_token], 0)

        output_tokens = []

        for _ in range(max_length):
            logits = self.decoder(context, decoder_input)

            next_token = tf.argmax(logits, axis=-1)

            # Append next token to output tokens
            output_tokens.append(next_token.numpy()[0, 0])  # Assuming batch size is 1

            if next_token == self.decoder.end_token:
                break

            decoder_input = next_token

        # Convert output tokens to words or sentences
        translated_sequence = self.decoder.tokens_to_text(output_tokens)

        return translated_sequence

class TranslationCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TranslationCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        source_text1 = "I'm an industrial engineer."
        result1 = model.translate([source_text1]) # 我是一个工业工程师

        source_text2 = "Hi, how are you today?"
        result2 = model.translate([source_text2])
        print(result2.numpy().decode())

        print(f"\nEpoch {epoch+1} Translation:")
        print(f"Source: {source_text1}")
        print(f"Translated: {result1.numpy().decode()}")
        print(f"Source: {source_text2}")
        print(f"Translated: {result2.numpy().decode()}")

#################
###           ###
###    Main   ###
###           ###
#################
def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

gpus = tf.config.experimental.list_physical_devices('GPU')
use_gpu = True

if not use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

if use_gpu and gpus:
    # Check if each GPU is available
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda/' # Change accordingly
    print("GPUs are available and configured.")
else:
    print("No GPUs are available. Running on CPU.")
    
data_loader = DataLoader()
data_loader.build("../data/train.csv", "../data/validation.csv")

UNITS = 256

model = Translator(Encoder(data_loader.en_context_text_processor, UNITS), Decoder(data_loader.zh_context_text_processor, UNITS))
model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_acc, masked_loss])

history = model.fit(
    data_loader.train_ds.repeat(), 
    epochs=50,
    steps_per_epoch = 100,
    validation_data=data_loader.val_ds,
    validation_steps = 20,
    callbacks=[TranslationCallback()]
)

# result = model.translate(["It's a tiny jungle party."]) # 就像这个小型的“丛林聚会
# print(result.numpy().decode())

# result = model.translate("Hi, how are you today?") # 就像这个小型的“丛林聚会
# print(result.numpy().decode())

# result = model.translate("I'm an industrial engineer.") # 我是一个工业工程师
# print(result)

# print(b'\xe7\x9a\x84'.decode('uft-8'))
