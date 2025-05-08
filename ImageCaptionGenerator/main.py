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

features = open("D:/documents/programming/projects/image captioning/ImageCaptionGenerator/features.p", "rb")

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

        