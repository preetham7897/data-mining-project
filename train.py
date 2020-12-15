import tensorflow as tf
import os
import logging
from utils import tokenize, text_retrieve, image_feature_retrieve, save_file, model_training, model_testing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():
    print()
    train_inp = text_retrieve('data/modified/captions/cleaned/train_image_id.txt')
    val_inp = text_retrieve('data/modified/captions/cleaned/val_image_id.txt')
    test_inp = text_retrieve('data/modified/captions/cleaned/test_image_id.txt')
    train_tar = text_retrieve('data/modified/captions/after-bpe/train_captions.txt')
    val_tar = text_retrieve('data/modified/captions/after-bpe/val_captions.txt')
    test_tar = text_retrieve('data/modified/captions/after-bpe/test_captions.txt')
    print('No. of original sentences in Training set: ', len(train_inp))
    print('No. of original sentences in Validation set: ', len(val_inp))
    print('No. of original sentences in Test set: ', len(test_inp))
    print()
    train_buffer_size = 20000
    val_buffer_size = 1500
    test_buffer_size = 1500
    tar_lang, train_tar, val_tar, test_tar = tokenize(train_tar[:train_buffer_size], val_tar[:val_buffer_size],
                                                      test_tar[:test_buffer_size])
    print('Target Vocabulary size: ', len(tar_lang.word_index) + 1)
    print()
    with tf.device('CPU:0'):
        train_inp = image_feature_retrieve(train_inp[:train_buffer_size])
        print('Training Input shape: ', train_inp.shape)
        print('Training Target shape: ', train_tar.shape)
        val_inp = image_feature_retrieve(val_inp[:val_buffer_size])
        print('Validation Input shape: ', val_inp.shape)
        print('Validation Target shape: ', val_tar.shape)
        test_inp = image_feature_retrieve(test_inp[:test_buffer_size])
        print('Testing Input shape: ', test_inp.shape)
        print('Testing Target shape: ', test_tar.shape)
    print()
    save_file(tar_lang.word_index, 'tar-word-index')
    save_file(tar_lang.index_word, 'tar-index-word')
    batch_size = 64
    parameters = {'tar_vocab_size': len(tar_lang.word_index) + 1, 'emb_size': 512, 'rnn_size': 512,
                  'batch_size': batch_size, 'epochs': 10, 'train_steps_per_epoch': len(train_inp) // batch_size,
                  'rate': 0.3, 'val_steps_per_epoch': len(val_inp) // batch_size,
                  'test_steps': len(test_inp) // batch_size}
    save_file(parameters, 'parameters')
    print()
    print('No. of Training steps per epoch: ', parameters['train_steps_per_epoch'])
    print('No. of Validation steps per epoch: ', parameters['val_steps_per_epoch'])
    print('No. of Testing steps: ', parameters['test_steps'])
    print()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inp, train_tar)).shuffle(len(train_inp))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_inp, val_tar)).shuffle(len(val_inp))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_inp, test_tar)).shuffle(len(test_inp))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    print('Model training started')
    print()
    #model_training(train_dataset, val_dataset)
    model_testing(test_dataset)

main()