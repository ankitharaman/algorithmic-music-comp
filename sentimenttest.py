# source: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
# the following code loads the sentiment analysis model, predicts the sentiment based on user text input
# then based on the sentiment, gives us a key signature corresponding to it

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
from music21 import key

# simply load the models from the website which should cache them
# zipping or pkling won't work as the model size is too big even after

# loading the cached model:

# tokenizer = AutoTokenizer.from_pretrained("./cardiffnlp/twitter-roberta-base-sentiment-latest")
# model = AutoModelForSequenceClassification.from_pretrained("./cardiffnlp/twitter-roberta-base-sentiment-latest")
# config = AutoConfig.from_pretrained("./cardiffnlp/twitter-roberta-base-sentiment-latest")

# this is how you would load the model from the website:

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

text = "Today is gonna be great!"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
# Print highest label and score
ranking = np.argsort(scores)
ranking = ranking[::-1]

my_label = config.id2label[ranking[0]]
my_score = scores[ranking[0]]
print(str(my_label) + " " + str(my_score))

# output: the highest ranked chosen label between positive, negative, and neutral and corresponding score

key_signatures = {
    # Flat keys (negative numbers)
    -7: {
        'major': key.Key('C-'),  # C-flat major
        'minor': key.Key('a-'),  # A-flat minor
    },
    -6: {
        'major': key.Key('G-'),  # G-flat major
        'minor': key.Key('e-'),  # E-flat minor
    },
    -5: {
        'major': key.Key('D-'),  # D-flat major
        'minor': key.Key('b-'),  # B-flat minor
    },
    -4: {
        'major': key.Key('A-'),  # A-flat major
        'minor': key.Key('f'),   # F minor
    },
    -3: {
        'major': key.Key('E-'),  # E-flat major
        'minor': key.Key('c'),   # C minor
    },
    -2: {
        'major': key.Key('B-'),  # B-flat major
        'minor': key.Key('g'),   # G minor
    },
    -1: {
        'major': key.Key('F'),   # F major
        'minor': key.Key('d'),   # D minor
    },
    
    # No sharps or flats
    0: {
        'major': key.Key('C'),   # C major
        'minor': key.Key('a'),   # A minor
    },
    
    # Sharp keys (positive numbers)
    1: {
        'major': key.Key('G'),   # G major
        'minor': key.Key('e'),   # E minor
    },
    2: {
        'major': key.Key('D'),   # D major
        'minor': key.Key('b'),   # B minor
    },
    3: {
        'major': key.Key('A'),   # A major
        'minor': key.Key('f#'),  # F-sharp minor
    },
    4: {
        'major': key.Key('E'),   # E major
        'minor': key.Key('c#'),  # C-sharp minor
    },
    5: {
        'major': key.Key('B'),   # B major
        'minor': key.Key('g#'),  # G-sharp minor
    },
    6: {
        'major': key.Key('F#'),  # F-sharp major
        'minor': key.Key('d#'),  # D-sharp minor
    },
    7: {
        'major': key.Key('C#'),  # C-sharp major
        'minor': key.Key('a#'),  # A-sharp minor
    }
}

def sentiment_to_key_and_tempo(label, score):
    """
    Map sentiment to a key signature, then get it as a music21 object.
    Also return a tempo based on sentiment.
    
    Parameters:
    label: 'negative', 'neutral', or 'positive'
    score: A value from 0 to 1 indicating intensity of sentiment
    
    Returns:
    A tuple containing the music21 key signature object and the tempo as an integer
    """

    # Define our ordered keys for each category (from least to most of that emotion)
    # Format: (key_signature_int, mode)
    negative_keys = [
        (2, 'minor'),   # B minor
        (1, 'minor'),   # E minor
        (-1, 'minor'),  # D minor
        (-2, 'minor'),  # G minor
        (-3, 'minor'),  # C minor
        (3, 'minor')    # F# minor - most 'negative' chord
    ]
    
    neutral_keys = [
        (-1, 'major'),  # F major
        (-2, 'major'),  # Bb major
        (-3, 'major'),  # Eb major
        (-4, 'major')   # Ab major
    ]
    
    positive_keys = [
        (0, 'major'),   # C major
        (1, 'major'),   # G major
        (2, 'major'),   # D major
        (3, 'major'),   # A major
        (5, 'major'),   # B major
        (4, 'major')    # E major - most 'positive' chord
    ]
    
    if label == 'negative':
        keys = negative_keys
        tempo = 45 + np.random.randint(40)
    elif label == 'neutral':
        keys = neutral_keys
        tempo = 75 + np.random.randint(30)
    else: 
        keys = positive_keys
        tempo = 95 + np.random.randint(70)
    
    # multiply score (0 to 1) by num of keys in category; use min to ensure we don't go out of bounds
    index = min(int(score * len(keys)), len(keys) - 1)
    
    num, mode = keys[index]
    return key_signatures[num][mode], tempo

my_key, tempo = sentiment_to_key_and_tempo(my_label, my_score)
print("Key: ", my_key)
print("Tempo: ", tempo)
