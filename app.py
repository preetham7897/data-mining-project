import os
from flask import Flask, request, render_template, send_from_directory
from model import Encoder, Decoder
import tensorflow as tf
import pickle
import logging
import sentencepiece as spm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def preprocess_image(file_name, model):
    w = tf.io.read_file(file_name)
    w = tf.image.decode_jpeg(w, channels=3)
    w = tf.image.resize(w, (299, 299))
    w = tf.keras.applications.inception_v3.preprocess_input(w)
    w = tf.convert_to_tensor([w])
    w = model(w)
    w = tf.reshape(w, [w.shape[0], -1, w.shape[3]])
    return w

def open_file(name):
    with open(name + '.pkl', 'rb') as f:
        d = pickle.load(f)
    f.close()
    return d

def predict(image_path, new_model):
    extracted_feature = preprocess_image(image_path, new_model)
    tar_word_index = open_file('/home/preetham/Documents/image-captioning/results/tar-word-index')
    tar_index_word = open_file('/home/preetham/Documents/image-captioning/results/tar-index-word')
    parameters = open_file('/home/preetham/Documents/image-captioning/results/parameters')
    encoder = Encoder(parameters['emb_size'], parameters['rate'])
    decoder = Decoder(parameters['emb_size'], parameters['tar_vocab_size'], parameters['rnn_size'], parameters['rate'])
    checkpoint_dir = '/home/preetham/Documents/image-captioning/results/training_checkpoints'
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    hidden = decoder.initialize_hidden_state(1, parameters['rnn_size'])
    enc_out = encoder(extracted_feature, False)
    dec_inp = tf.expand_dims([tar_word_index['<s>']], 0)
    caption = []
    for i in range(1, 100):
        prediction, hidden = decoder(dec_inp, hidden, enc_out, False)
        predicted_id = tf.argmax(prediction[0]).numpy()
        if tar_index_word[predicted_id] != '</s>':
            caption.append(tar_index_word[predicted_id])
        else:
            break
        dec_inp = tf.expand_dims([predicted_id], 0)
    sp = spm.SentencePieceProcessor()
    sp.load('/home/preetham/Documents/image-captioning/results/en.model')
    caption = sp.DecodePieces(caption)
    caption = caption.replace('▁', ' ')
    return caption

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    upload = request.files['file']
    print(upload)
    print("{} is the file name".format(upload.filename))
    filename = upload.filename
    # This is to verify files are supported
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg"):
        print("File supported moving on...")
    else:
        print("File not supported")
        return render_template("Error.html", message="File uploaded is not supported... only .jpeg file format accepted")
    dir = "images"
    n = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    print('No of files in images folder :')
    print(n)
    n1 = str(n)+'.jpg'
    destination = "/".join([target, n1])
    print("Accept incoming file:", filename)
    print("Save it to:", destination)
    upload.save(destination)
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_model = tf.keras.Model(model.input, model.layers[-1].output)
    new_model.trainable = False
    caption = predict(destination, new_model)
    print(caption)
    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=n1, c=caption)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(debug=True)
