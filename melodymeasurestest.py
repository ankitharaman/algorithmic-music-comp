import music21
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('harmony_model.h5')

# Sample melody in C major - a bit longer and more varied
# Format: [(scale_degree, accidental, duration), ...]
test_melody = [
    # Measure 1
    (1, 0, 1.0),   # C - scale degree 1, quarter note
    (2, 0, 1.0),   # D - scale degree 2, quarter note
    (3, 0, 1.0),   # E - scale degree 3, quarter note
    (5, 0, 1.0),   # G - scale degree 5, quarter note
    
    # Measure 2
    (5, 0, 0.5),   # G - scale degree 5, eighth note
    (6, 0, 0.5),   # A - scale degree 6, eighth note
    (5, 0, 1.0),   # G - scale degree 5, quarter note
    (4, 0, 1.0),   # F - scale degree 4, quarter note
    (3, 0, 1.0),   # E - scale degree 3, quarter note
    
    # Measure 3
    (4, 0, 1.0),   # F - scale degree 4, quarter note
    (3, 0, 1.0),   # E - scale degree 3, quarter note
    (2, 0, 1.0),   # D - scale degree 2, quarter note
    (1, 0, 1.0),   # C - scale degree 1, quarter note
]

# Create key
key = music21.key.Key('C')  # C major

def predict_harmony(model, melody_measure, key):
    # Preprocess melody
    processed_melody = []
    for scale_deg, acc, dur in melody_measure:
        if scale_deg is None:  # Rest
            processed_melody.append([-1, 1, dur])  # -1 for rest scale degree
        else:
            normalized_scale_deg = scale_deg - 1  # Convert to 0-6
            processed_melody.append([normalized_scale_deg, acc + 1, dur])  # Shift accidental to 0,1,2
    
    # Reshape for model input
    model_input = np.array([processed_melody])
    
    # Pad input if needed
    padded_input = keras.preprocessing.sequence.pad_sequences(
        model_input, padding='post', dtype='float32'
    )
    
    # Get predictions
    predictions = model.predict(padded_input)
    
    # Extract predicted values and ensure they're valid
    first_scale = np.argmax(predictions[0][0]) + 1  # Convert back to 1-7
    first_acc = np.argmax(predictions[1][0]) - 1    # Convert back to -1,0,1
    second_scale = np.argmax(predictions[2][0]) + 1
    second_acc = np.argmax(predictions[3][0]) - 1
    
    # Ensure scale degrees are valid (1-7)
    first_scale = max(1, min(7, first_scale))
    second_scale = max(1, min(7, second_scale))
    
    # Ensure accidentals are valid (-1, 0, 1)
    first_acc = max(-1, min(1, first_acc))
    second_acc = max(-1, min(1, second_acc))
    
    print(f"Predicted harmony:")
    print(f"First chord: Scale degree {first_scale} with accidental {first_acc}")
    print(f"Second chord: Scale degree {second_scale} with accidental {second_acc}")
    
    # Create actual notes/chords
    first_chord = create_note_or_chord(key, first_scale, first_acc)
    second_chord = create_note_or_chord(key, second_scale, second_acc)
    
    return first_chord, second_chord

def create_note_or_chord(key, scale_degree, accidental, create_chord=True):
    """Convert scale degree to a note or chord."""
    try:
        scale = key.getScale()
        
        # Get the pitch for the scale degree
        # This is the problematic line - let's fix it
        if scale_degree < 1 or scale_degree > 7:
            print(f"Invalid scale degree: {scale_degree}, defaulting to 1")
            scale_degree = 1
            
        pitch = scale.pitchFromDegree(scale_degree)
        
        # Apply accidental
        if accidental == -1:
            pitch = pitch.transpose(-1)
        elif accidental == 1:
            pitch = pitch.transpose(1)
        
        if create_chord:
            # Create chord based on root pitch
            if key.mode == 'major':
                # Major key harmony
                if scale_degree in [1, 4, 5]:  # Major chords (I, IV, V)
                    chord = music21.chord.Chord([
                        pitch.nameWithOctave, 
                        pitch.transpose(4).nameWithOctave, 
                        pitch.transpose(7).nameWithOctave
                    ])
                    return chord
                elif scale_degree in [2, 3, 6]:  # Minor chords (ii, iii, vi)
                    chord = music21.chord.Chord([
                        pitch.nameWithOctave, 
                        pitch.transpose(3).nameWithOctave, 
                        pitch.transpose(7).nameWithOctave
                    ])
                    return chord
                elif scale_degree == 7:  # Diminished chord (vii°)
                    chord = music21.chord.Chord([
                        pitch.nameWithOctave, 
                        pitch.transpose(3).nameWithOctave, 
                        pitch.transpose(6).nameWithOctave
                    ])
                    return chord
            else:
                # Minor key harmony
                if scale_degree in [1, 4]:  # Minor chords (i, iv)
                    chord = music21.chord.Chord([
                        pitch.nameWithOctave, 
                        pitch.transpose(3).nameWithOctave, 
                        pitch.transpose(7).nameWithOctave
                    ])
                    return chord
                elif scale_degree in [3, 5, 6]:  # Major chords (III, V, VI)
                    chord = music21.chord.Chord([
                        pitch.nameWithOctave, 
                        pitch.transpose(4).nameWithOctave, 
                        pitch.transpose(7).nameWithOctave
                    ])
                    return chord
                elif scale_degree in [2, 7]:  # Diminished chords (ii°, vii°)
                    chord = music21.chord.Chord([
                        pitch.nameWithOctave, 
                        pitch.transpose(3).nameWithOctave, 
                        pitch.transpose(6).nameWithOctave
                    ])
                    return chord
        else:
            # Just return the note
            return music21.note.Note(pitch)
    except Exception as e:
        print(f"Error creating chord: {e}")
        # Return a C major chord as fallback
        return music21.chord.Chord(['C4', 'E4', 'G4'])

# Create a function to process each measure separately
def predict_harmony_for_measures(model, melody, key):
    # Split melody into measures (assuming 4/4 time signature)
    measures = []
    current_measure = []
    current_duration = 0
    
    for note in melody:
        if current_duration >= 4.0:  # Start a new measure when we reach 4 beats
            measures.append(current_measure)
            current_measure = []
            current_duration = 0
        
        current_measure.append(note)
        current_duration += note[2]  # Add note duration
    
    # Add the last measure if it has notes
    if current_measure:
        measures.append(current_measure)
    
    # Process each measure
    predicted_chords = []
    for measure in measures:
        first_chord, second_chord = predict_harmony(model, measure, key)
        predicted_chords.append((first_chord, second_chord))
    
    return predicted_chords, measures

# Create a score with the results
def create_score_with_harmony(melody, predicted_chords, measures, key):
    score = music21.stream.Score()
    
    # Add melody
    melody_part = music21.stream.Part()
    melody_part.id = 'Melody'
    
    # Add harmony
    harmony_part = music21.stream.Part()
    harmony_part.id = 'Harmony'
    
    # Add time signature
    time_sig = music21.meter.TimeSignature('4/4')
    melody_part.append(time_sig)
    harmony_part.append(time_sig)
    
    # Add key signature
    melody_part.append(key)
    harmony_part.append(key)
    
    # Process each measure
    for measure_idx, (measure_notes, (first_chord, second_chord)) in enumerate(zip(measures, predicted_chords)):
        # Create melody measure
        m_melody = music21.stream.Measure(number=measure_idx+1)
        
        # Add melody notes
        current_offset = 0
        for scale_deg, acc, dur in measure_notes:
            pitch = key.getScale().pitchFromDegree(scale_deg)
            if acc == -1:
                pitch = pitch.transpose(-1)
            elif acc == 1:
                pitch = pitch.transpose(1)
            note = music21.note.Note(pitch, quarterLength=dur)
            note.offset = current_offset
            m_melody.append(note)
            current_offset += dur
        
        # Create harmony measure
        m_harmony = music21.stream.Measure(number=measure_idx+1)
        
        # Add first chord at beat 1
        first_chord.offset = 0.0
        first_chord.quarterLength = 2.0
        m_harmony.append(first_chord)
        
        # Add second chord at beat 3
        second_chord.offset = 2.0
        second_chord.quarterLength = 2.0
        m_harmony.append(second_chord)
        
        # Add measures to parts
        melody_part.append(m_melody)
        harmony_part.append(m_harmony)
    
    # Add both parts to score
    score.append(melody_part)
    score.append(harmony_part)
    
    return score

# Predict harmony for each measure
predicted_chords, measures = predict_harmony_for_measures(model, test_melody, key)

# Create the score
score = create_score_with_harmony(test_melody, predicted_chords, measures, key)

# Show and save the score
score.write('musicxml', 'test_melody_with_harmony.xml')
score.show()
