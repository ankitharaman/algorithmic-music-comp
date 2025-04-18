#!/usr/bin/env python

from music21 import *

postive_measures = []
negative_measures = []
melody = []

constructed_harmony = []

def evaluate_measure_pair(k, melody_measure, harmony_measure):
    """
    k is the key
    melody_measure is an array of tuples of the form (scale degree, accidental,
                                                      duration)
    harmony_measure is an array of music21.chord.Chord objects
    """

    num_melody_notes = len(melody_measure)

    tone_match_score = 0
    chord_progression_score = 0

    for i in range(num_melody_notes):
        note = melody_measure[i]
        harmony_chord = harmony_measure[0]
        if i > num_melody_notes // 2:
            harmony_chord = harmony_measure[1]

        pitch = key.getScale().pitchFromDegree(note[0])
        if pitch in harmony_chord.pitches:
            tone_match_score += 1
        
    if len(constructed_harmony) > 0:
        prev_chord = constructed_harmony[-1]


    score = tone_match_score + chord_progression_score*num_melody_notes
    return score/num_melody_notes
