import tensorflow as tf
import pickle
import time
import matplotlib.pyplot as plt
from model import Encoder, Decoder
import numpy as np
import re
import sentencepiece as spm
import pandas as pd

def text_retrieve(name):
    with open('/home/preetham/Documents/image-captioning/' + name, 'r',
              encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text.split('\n')

def tokenize(train, val, test):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(train)
    train_tensor = lang_tokenizer.texts_to_sequences(train)
    train_tensor = tf.keras.preprocessing.sequence.pad_sequences(train_tensor, padding='post')
    val_tensor = lang_tokenizer.texts_to_sequences(val)
    val_tensor = tf.keras.preprocessing.sequence.pad_sequences(val_tensor, padding='post')
    test_tensor = lang_tokenizer.texts_to_sequences(test)
    test_tensor = tf.keras.preprocessing.sequence.pad_sequences(test_tensor, padding='post')
    return lang_tokenizer, train_tensor, val_tensor, test_tensor

def open_file(name):
    with open(name + '.pkl', 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def save_file(d, name):
    loc_to = '/home/preetham/Documents/image-captioning/results/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    print(name + ' saved successfully')
    f.close()

def image_feature_retrieve(file_names):
    features = []
    for i in file_names:
        d = open_file('/home/preetham/Documents/image-captioning/data/modified/images/' + i)
        features.append(d)
    with tf.device('CPU:0'):
        features = tf.convert_to_tensor(features)
        features = tf.reshape(features, [features.shape[0], features.shape[2], features.shape[3]])
        return features

def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, tar, encoder, decoder, optimizer, tar_word_index, batch_size, hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_out = encoder(inp, True)
        dec_inp = tf.expand_dims([tar_word_index['<s>']] * batch_size, 1)
        for i in range(1, tar.shape[1]):
            prediction, hidden = decoder(dec_inp, hidden, enc_out, True)
            loss += loss_function(tar[:, i], prediction)
            dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(batch_loss)

def validation_step(inp, tar, encoder, decoder, tar_word_index, batch_size, hidden):
    loss = 0
    enc_out = encoder(inp, False)
    dec_inp = tf.expand_dims([tar_word_index['<s>']] * batch_size, 1)
    for i in range(1, tar.shape[1]):
        prediction, hidden = decoder(dec_inp, hidden, enc_out, False)
        loss += loss_function(tar[:, i], prediction)
        dec_inp = tf.expand_dims(tar[:, i], 1)
    batch_loss = loss / tar.shape[1]
    val_loss(batch_loss)

def model_training(train_dataset, val_dataset):
    global train_loss, val_loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    parameters = open_file('/home/preetham/Documents/image-captioning/results/parameters')
    encoder = Encoder(parameters['emb_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    optimizer = tf.keras.optimizers.Adam()
    checkpoint_dir = '/home/preetham/Documents/image-captioning/results/training_checkpoints'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)
    tar_word_index = open_file('/home/preetham/Documents/image-captioning/results/tar-word-index')
    split_df = pd.DataFrame(columns=['steps', 'train_loss', 'val_loss'])
    step = 0
    best_val_loss = None
    checkpoint_count = 0
    for epoch in range(parameters['epochs']):
        epoch_start = time.time()
        hidden = decoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
        train_loss.reset_states()
        val_loss.reset_states()
        for (batch, (inp, tar)) in enumerate(train_dataset.take(parameters['train_steps_per_epoch'])):
            batch_start = time.time()
            train_step(inp, tar, encoder, decoder, optimizer, tar_word_index, parameters['batch_size'], hidden)
            batch_end = time.time()
            if batch % 10 == 0:
                print('Epoch=' + str(epoch + 1) + ', Batch=' + str(batch) + ', Training Loss=' +
                      str(round(train_loss.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        for (batch, (inp, tar)) in enumerate(val_dataset.take(parameters['val_steps_per_epoch'])):
            batch_start = time.time()
            validation_step(inp, tar, encoder, decoder, tar_word_index, parameters['batch_size'], hidden)
            batch_end = time.time()
            if batch % 10 == 0:
                print('Epoch=' + str(epoch + 1) + ', Batch=' + str(batch) + ', Validation Loss=' +
                      str(round(val_loss.result().numpy(), 3)) + ', Time taken=' +
                      str(round(batch_end - batch_start, 3)) + ' sec')
        d = {'steps': int(step), 'train_loss': train_loss.result().numpy(), 'val_loss': val_loss.result().numpy()}
        split_df = split_df.append(d, ignore_index=True)
        split_df.to_csv('/home/preetham/Documents/image-captioning/results/split_steps.csv',
                        index=False)
        if best_val_loss is None:
            checkpoint_count = 0
            best_val_loss = round(val_loss.result().numpy(), 3)
            manager.save()
            print('Checkpoint saved')
            print()
        elif best_val_loss > round(val_loss.result().numpy(), 3):
            checkpoint_count = 0
            print('Best Validation Loss changed from ' + str(best_val_loss) + ' to ' +
                  str(round(val_loss.result().numpy(), 3)))
            best_val_loss = round(val_loss.result().numpy(), 3)
            manager.save()
            print('Checkpoint saved')
            print()
        elif checkpoint_count <= 4:
            checkpoint_count += 1
            print('Best Validation Loss did not improve')
            print('Checkpoint not saved')
            print()
        else:
            print('Model did not improve after 4th time. Model stopped from training further.')
            print()
            break
        epoch_end = time.time()
        print('Epoch=' + str(epoch + 1) + ', Training Loss=' + str(round(train_loss.result().numpy(), 3)) +
              ', Validation Loss=' + str(round(val_loss.result().numpy(), 3)) + ' , Time taken=' +
              str(round(epoch_end-epoch_start, 3)))
        print()

def model_testing(test_dataset):
    global val_loss
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    parameters = open_file('/home/preetham/Documents/image-captioning/results/parameters')
    val_loss.reset_states()
    encoder = Encoder(parameters['emb_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    checkpoint_dir = '/home/preetham/Documents/image-captioning/results/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    tar_word_index = open_file('/home/preetham/Documents/image-captioning/results/tar-word-index')
    hidden = decoder.initialize_hidden_state(parameters['batch_size'], parameters['rnn_size'])
    for (batch, (inp, tar)) in enumerate(test_dataset.take(parameters['test_steps'])):
            validation_step(inp, tar, encoder, decoder, tar_word_index, parameters['batch_size'], hidden)
    print('Test Loss=', round(val_loss.result().numpy(), 3))
    print()