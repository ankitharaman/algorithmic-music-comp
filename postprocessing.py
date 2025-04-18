#!/usr/bin/env python

from music21 import *
import markov_chain

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
        scale_deg1 = k.getScaleDegreeFromPitch(prev_chord.root())
        curr_scale_deg = k.getScaleDegreeFromPitch(harmony_measure[0].root())
        chord_progression_score += markov_chain.M2[scale_deg1 - 1][curr_scale_deg - 1]

    prev_chord = harmony_measure[0]
    scale_deg1 = k.getScaleDegreeFromPitch(prev_chord.root())
    curr_scale_deg = k.getScaleDegreeFromPitch(harmony_measure[1].root())
    chord_progression_score += markov_chain.M2[scale_deg1 - 1][curr_scale_deg - 1]


    score = tone_match_score + chord_progression_score*num_melody_notes//2
    return score/num_melody_notes

def construct_harmony(k, melody_measures, pos_harmony_measures, 
                      neg_harmony_measures, pos_sent_score, neg_sent_score):
    for i, melody_measure in enumerate(melody_measures):
        pos_score = evaluate_measure_pair(k, melody_measure, pos_harmony_measures[i])
        pos_score = (pos_score + (pos_sent_score * 0.5)) / 1.5
        neg_score = evaluate_measure_pair(k, melody_measure, neg_harmony_measures[i])
        neg_score = (neg_score + (neg_sent_score * 0.5)) / 1.5

        if pos_score > neg_score:
            constructed_harmony.append(pos_harmony_measures[i])
        else:
            constructed_harmony.append(neg_harmony_measures[i])