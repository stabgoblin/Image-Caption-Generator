import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# ---------- Feature Extraction ----------
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        raise FileNotFoundError("ERROR: Couldn't open image! Make sure the image path and extension are correct.")
    
    image = image.resize((299, 299))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[..., :3]  # Convert RGBA to RGB

    image = np.expand_dims(image, axis=0)
    image = image / 127.5 - 1.0  # Xception expects inputs in range [-1, 1]
    feature = model.predict(image, verbose=0)
    return feature

# ---------- Word Mapping ----------
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# ---------- Generate Caption ----------
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    final_caption = in_text.split()
    if final_caption[0] == 'start':
        final_caption = final_caption[1:]
    if final_caption and final_caption[-1] == 'end':
        final_caption = final_caption[:-1]
    return ' '.join(final_caption)

# ---------- Define the Model ----------
def define_model(vocab_size, max_length):
    # Feature extractor (image)
    inputs1 = Input(shape=(2048,), name='image_input')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence processor (text)
    inputs2 = Input(shape=(max_length,), name='text_input')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (merge and predict next word)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ---------- Main Execution ----------
if __name__ == "__main__":
    # Command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help="Path to input image")
    args = vars(ap.parse_args())
    img_path = args['image']

    # Constants
    max_length = 32
    tokenizer_path = "D:/documents/programming/projects/image captioning/ImageCaptionGenerator/tokenizer.p"
    weights_path = "D:/documents/programming/projects/image captioning/ImageCaptionGenerator/models/model_9.h5"

    # Load tokenizer
    tokenizer = load(open(tokenizer_path, "rb"))
    vocab_size = len(tokenizer.word_index) + 1

    # Load model architecture & weights
    model = define_model(vocab_size, max_length)
    model.load_weights(weights_path)

    # Load CNN feature extractor
    xception_model = Xception(include_top=False, pooling="avg")

    # Extract features & generate description
    photo = extract_features(img_path, xception_model)
    img = Image.open(img_path)
    description = generate_desc(model, tokenizer, photo, max_length)

    # Output
    print("\nGenerated Caption:")
    print(description)
    plt.imshow(img)
    plt.axis('off')
    plt.title(description)
    plt.show()

# python test.py --image Flicker8k_Dataset/1859941832_7faf6e5fa9.jpg