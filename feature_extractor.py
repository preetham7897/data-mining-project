import tensorflow as tf
import json
import os
import unicodedata
import re
import logging
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def open_json_file(file_name):
    path = '/home/preetham/Documents/image-captioning/data/original/annotations/'
    with open(path + file_name, 'r') as f:
        captions = json.load(f)
    return captions

def save_file(d, name):
    loc_to = '/home/preetham/Documents/image-captioning/data/modified/images/'
    with open(loc_to + name + '.pkl', 'wb') as f:
        pickle.dump(d, f, protocol=2)
    f.close()

def file_name_retrieve():
    path = '/home/preetham/Documents/image-captioning/data/'
    file_names = [os.path.join(path + 'original/train2014/', f) for f in os.listdir(path + 'original/train2014/') if
                  os.path.isfile(os.path.join(path + 'original/train2014/', f))]
    return file_names

def remove_html_markup(s):
    tag = False
    quote = False
    out = ""
    for c in s:
        if c == '<' and not quote:
            tag = True
        elif c == '>' and not quote:
            tag = False
        elif (c == '"' or c == "'") and tag:
            quote = not quote
        elif not tag:
            out = out + c
    return out

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = remove_html_markup(w)
    w = w.lower().strip()
    if w == '':
        return 0
    else:
        w = unicode_to_ascii(w)
        w = re.sub(r"[^-!$&(),./%0-9:;?a-z'\"]+", " ", w)
        w = re.sub(r'(\d)th', r'\1 th', w, flags=re.I)
        w = re.sub(r'(\d)st', r'\1 st', w, flags=re.I)
        w = re.sub(r'(\d)rd', r'\1 rd', w, flags=re.I)
        w = re.sub(r'(\d)nd', r'\1 nd', w, flags=re.I)
        punc = list("-!$&(),./%:;?¿¡€'")
        for i in punc:
            w = w.replace(i, " "+i+" ")
        w = w.strip()
        w = re.sub(r'\s+', ' ', w)
        return w

def preprocess_image(file_name, model):
    w = tf.io.read_file(file_name)
    w = tf.image.decode_jpeg(w, channels=3)
    w = tf.image.resize(w, (299, 299))
    w = tf.keras.applications.inception_v3.preprocess_input(w)
    w = tf.convert_to_tensor([w])
    w = model(w)
    w = tf.reshape(w, [w.shape[0], -1, w.shape[3]])
    return w

def create_dataset(file_names, model):
    train_captions = open_json_file('captions_train2014.json')
    annotations = pd.DataFrame(train_captions['annotations'])
    annotations = shuffle(annotations)
    image_id, captions = [], []
    image_count = 0
    for i in range(len(annotations)):
        if annotations['image_id'].iloc[i] in image_id:
            continue
        image_path = '/home/preetham/Documents/image-captioning/data/original/train2014/COCO_train2014_' + '%012d.jpg' \
                     % int(annotations['image_id'][i])
        if image_path in file_names:
            extracted_features = preprocess_image(image_path, model)
            save_file(extracted_features, str(annotations['image_id'].iloc[i]))
            new_annotations = annotations[annotations['image_id'] == annotations['image_id'].iloc[i]]
            for j in range(len(new_annotations)):
                caption = preprocess_sentence(new_annotations['caption'].iloc[j])
                if caption == 0:
                    continue
                captions.append(caption)
                image_id.append(str(new_annotations['image_id'].iloc[j]))
            image_count += 1
        if image_count % 100 == 0:
            print('No. of images processed: ', image_count)
        if image_count == 30000:
            break
    return image_id, captions

def lines_to_text(lines, sep):
    text = ''
    for i in range(len(lines)):
        if i == len(lines) - 1:
            text += str(lines[i])
        else:
            text += str(lines[i]) + sep
    return text

def dataset_save(lines, name):
    text = lines_to_text(lines, '\n')
    f = open('/home/preetham/Documents/image-captioning/data/modified/'+name, 'w', encoding='utf-8')
    f.write(text)
    f.close()

def main():
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_model = tf.keras.Model(model.input, model.layers[-1].output)
    new_model.trainable = False
    print()
    file_names = file_name_retrieve()
    image_id, captions = create_dataset(file_names, new_model)
    print('Dataset size: ', len(image_id))
    print()
    train_image_id, val_image_id, train_captions, val_captions = train_test_split(image_id, captions, test_size=0.2)
    val_image_id, test_image_id, val_captions, test_captions = train_test_split(val_image_id, val_captions,
                                                                                test_size=0.5)
    print('Training set size: ', len(train_image_id))
    print('Validation set size: ', len(val_image_id))
    print('Testing set size: ', len(test_image_id))
    print()
    dataset_save(train_image_id, 'train_image_id.txt')
    dataset_save(train_captions, 'train_captions.txt')
    dataset_save(val_image_id, 'val_image_id.txt')
    dataset_save(val_captions, 'val_captions.txt')
    dataset_save(test_image_id, 'test_image_id.txt')
    dataset_save(test_captions, 'test_captions.txt')

main()