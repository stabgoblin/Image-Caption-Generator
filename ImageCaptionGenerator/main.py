import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, get_file
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# small library for seeing the progress of loops.
from tqdm import tqdm
tqdm.pandas()

# Loading a text file into memory
def load_doc(filename):
    # Opening the file as read only
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# get all imgs with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions ={}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions:
            descriptions[img[:-2]] = [ caption ]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

#Data cleaning- lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans('','',string.punctuation)
    for img,caps in captions.items():
        for i,img_caption in enumerate(caps):

            img_caption.replace("-"," ")
            desc = img_caption.split()

            #converts to lowercase
            desc = [word.lower() for word in desc]
            #remove punctuation from each token
            desc = [word.translate(table) for word in desc]
            #remove hanging 's and a 
            desc = [word for word in desc if(len(word)>1)]
            #remove tokens with numbers in them
            desc = [word for word in desc if(word.isalpha())]
            #convert back to string

            img_caption = ' '.join(desc)
            captions[img][i]= img_caption
    return captions

def text_vocabulary(descriptions):
    # build vocabulary of all unique words
    vocab = set()

    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]

    return vocab

#All descriptions in one file 
def save_descriptions(descriptions, filename):
    # Ensure the folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the descriptions
    with open(filename, "w", encoding="utf-8") as file:
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                file.write(f"{key}\t{desc}\n")


dataset_text =  "D:/documents/programming/projects/image captioning/ImageCaptionGenerator/Flickr8k_text"
dataset_images = "D:/documents/programming/projects/image captioning/ImageCaptionGenerator/Flicker8k_Dataset"
#
#filename = dataset_text + "/Flickr8k.token.txt"
#descriptions = all_img_captions(filename)
#print("length of descriptions = ", len(descriptions))
#
#clean_descriptions = cleaning_text(descriptions)
#vocabulary = text_vocabulary(clean_descriptions)
#print("length of vocabulary = ",len(vocabulary))
#
#save_descriptions(clean_descriptions,"D:/documents/programming/projects/image captioning/ImageCaptionGenerator/descriptions.txt")

#def download_with_retry(url,filename,max_retries = 3):
#    for attempt in range(max_retries):
#        try:
#            return get_file(filename,url)
#        except Exception as e:
#            if attempt == max_retries -1:
#                raise e
#            print("Download failed")
#            time.sleep(3)
#
#weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
#weights_path = download_with_retry(weights_url,'xception_weights.h5')
#model = Xception(include_top = False, pooling = "avg", weights = weights_path)
#
#def extract_features(directory):
#    features = {}
#    valid_images = [".jpg",".jpeg",".png"]
#    for img in tqdm(os.listdir(directory)):
#        ext = os.path.splitext(img)[1].lower()
#        if ext not in valid_images:
#            continue
#        filename = directory + "/" + img
#        image = Image.open(filename)
#        image = image.resize((299,299))
#        image = np.expand_dims(image,axis = 0)
#        image = image / 127.5
#        image = image - 1.0
#        feature = model.predict(image)
#        features[img] = feature
#
#    return features
#
#features = extract_features(dataset_images)
#with open("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/features.p", "wb") as file:
#    dump(features, file)

features = load(open("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/features.p", "rb"))

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[::-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images,photo))]
    return photos_present

def load_clean_descriptions(filename,photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions

def load_features(photos):
    all_features = load(open("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/features.p","rb"))
    features = {k: all_features[k] for k in photos if k in all_features and k.strip() != ''}
    print(features)
    return features

filename = 'D:/documents/programming/projects/image captioning/ImageCaptionGenerator/Flickr8k_text/Flickr_8k.trainImages.txt'

train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/descriptions.txt",train_imgs)
train_features = load_features(train_imgs)

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)

dump(tokenizer, open("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/tokenizer.p",'wb'))

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_length = max_length(train_descriptions)
print(max_length)

def data_generator(descriptions, features, tokenizer, max_length):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]
    
    # Define the output signature for the generator
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    
    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    return dataset.batch(32)

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

dataset = data_generator(train_descriptions, features, tokenizer, max_length)
for (a,b) in dataset.take(1):
    print(a['input_1'].shape, a['input_2'].shape,b.shape)
    break

def define_model(vocab_size, max_length):
    #CNN model from 2048 nodes to 256 nodes
    inputs1 = Input(shape = (2048,), name = 'input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation = 'relu')(fe1)

    #lstm sequence model
    inputs2 = Input(shape = (max_length,),name = 'input_2')
    se1 = Embedding(vocab_size,256, mask_zero = True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2,se3])
    decoder2 = Dense(256, activation = 'relu')(decoder1)
    outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
    model = Model(inputs = [inputs1, inputs2], outputs = outputs)

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')    
    print(model.summary())
    return model

model = define_model(vocab_size, max_length)

os.mkdir('D:/documents/programming/projects/image captioning/ImageCaptionGenerator/models')
for i in range(10):
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit(dataset, epochs = 10, steps_per_epoch = 5, verbose = 1)
    model.save("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/models/model_" + str(i) + ".h5")
