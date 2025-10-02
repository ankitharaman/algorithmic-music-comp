# Sentiment to Sound - Algorithmic Music Composition

This project generates music from text input by combining data structures, sentiment analysis, and a neural network–based harmony predictor.

Given a string of text, the algorithm outputs a melody (mapped from data structure operations) and harmony (predicted using a trained model), creating a structured piece of music.

## How It Works

### User Input:
The user provides a string of text.

### Get Sentiment:
The sentiment of the text determines the key signature and tempo.

### Generate Melody:
Melodies are generated as aural representations of data structure operations:

- Circular Buffer
- Hash Table
- Red-Black Tree

### Predict Harmony:
A pre-trained neural network predicts harmonies for the generated melody.

## Install Dependencies:
```
pip install -r requirements.txt
```

Includes: music21, numpy, tensorflow / keras

### Usage:
1. Preprocess Training Data
- The main.py file preprocesses positive and negative training examples (classical scores in MusicXML format)
  ```
  python main.py --preprocess
  ```
- These will generate positive.pkl & negative.pkl, which contain serialized melody-harmony pairs for model training
2. Generate Music From Text
- Running ``` python predict_harmony.py ``` generates a melody and harmony from user input
  - This file will output a MusicXML file and play back the result as MIDI
  - To use your own input string, edit the ``` user_message ``` in ``` predict_harmony.py ```

