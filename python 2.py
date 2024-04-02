import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample parallel corpus
source_sentences = [
    'I am hungry',
    'He is a doctor',
    'She likes singing'
]
target_sentences = [
    'Je suis affamé',
    'Il est médecin',
    'Elle aime chanter'
]

# Tokenize source and target sentences
source_tokenizer = Tokenizer()
source_tokenizer.fit_on_texts(source_sentences)
source_sequences = source_tokenizer.texts_to_sequences(source_sentences)

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_sentences)
target_sequences = target_tokenizer.texts_to_sequences(target_sentences)

# Pad sequences to ensure uniform length
max_source_length = max([len(seq) for seq in source_sequences])
max_target_length = max([len(seq) for seq in target_sequences])
source_sequences_padded = pad_sequences(source_sequences, maxlen=max_source_length, padding='post')
target_sequences_padded = pad_sequences(target_sequences, maxlen=max_target_length, padding='post')

# Define the model architecture
embedding_dim = 16
units = 32
source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

# Encoder
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(source_vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(units, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit([source_sequences_padded, target_sequences_padded[:, :-1]], target_sequences_padded[:, 1:],
          batch_size=32, epochs=50, validation_split=0.2)

# Inference
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(units,))
decoder_state_input_c = tf.keras.layers.Input(shape=(units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_char
        if (sampled_char == '<end>' or
           len(decoded_sentence) > max_target_length):
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
    return decoded_sentence

# Translate a sentence
input_sentence = 'I am hungry'
input_sequence = source_tokenizer.texts_to_sequences([input_sentence])
input_sequence_padded = pad_sequences(input_sequence, maxlen=max_source_length, padding='post')
translation = translate_sentence(input_sequence_padded)
print('Input sentence:', input_sentence)
print('Translated sentence:', translation)
