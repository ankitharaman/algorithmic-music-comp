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
pip install -r requirements.txt

