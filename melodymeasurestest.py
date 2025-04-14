import music21
import numpy as np
from tensorflow import keras


# Load the trained model
model = keras.models.load_model('harmony_model.h5')

# Sample melody in C major - a bit longer and more varied
# Format: [(scale_degree, accidental, duration), ...]
# test_melody = [
#     # Measure 1
#     (1, 0, 1.0),   # C - scale degree 1, quarter note
#     (2, 0, 1.0),   # D - scale degree 2, quarter note
#     (3, 0, 1.0),   # E - scale degree 3, quarter note
#     (5, 0, 1.0),   # G - scale degree 5, quarter note
    
#     # Measure 2
#     (5, 0, 0.5),   # G - scale degree 5, eighth note
#     (6, 0, 0.5),   # A - scale degree 6, eighth note
#     (5, 0, 1.0),   # G - scale degree 5, quarter note
#     (4, 0, 1.0),   # F - scale degree 4, quarter note
#     (3, 0, 1.0),   # E - scale degree 3, quarter note
    
#     # Measure 3
#     (4, 0, 1.0),   # F - scale degree 4, quarter note
#     (3, 0, 1.0),   # E - scale degree 3, quarter note
#     (2, 0, 1.0),   # D - scale degree 2, quarter note
#     (1, 0, 1.0),   # C - scale degree 1, quarter note
# ]

test_melody = [
    # Measure 1: Suggests I to V7 with passing tones
    (1, 0, 1.0),   # C - suggests I chord
    (7, -1, 0.5),  # B♭ - suggests V7/IV
    (6, 0, 0.5),   # A - suggests vi or IV
    (4, 0, 1.0),   # F - suggests IV chord
    (5, 0, 1.0),   # G - suggests V chord
    
    # Measure 2: Suggests ii-V-I progression
    (2, 0, 0.5),   # D - suggests ii chord
    (4, 0, 0.5),   # F - part of ii chord
    (3, 0, 0.5),   # E - passing tone
    (2, 0, 0.5),   # D - suggests ii chord
    (7, 0, 1.0),   # B - suggests V7 chord
    (1, 0, 1.0),   # C - suggests I chord
    
    # Measure 3: Suggests vi-ii-V pattern
    (6, 0, 1.0),   # A - suggests vi chord
    (2, 0, 1.0),   # D - suggests ii chord
    (5, 0, 1.0),   # G - suggests V chord
    (7, 0, 1.0),   # B - part of V7 chord
]

# Create key
key = music21.key.Key('C')  # C major

def predict_harmony(model, melody_measure, key, temperature=1.0, use_sampling=True):
    """Predict harmony with optional temperature control and sampling."""
    # Preprocess melody
    processed_melody = []
    for scale_deg, acc, dur in melody_measure:
        if scale_deg is None:  # Rest
            processed_melody.append([-1, 1, dur])  # Use -1 for rest scale degree
        else:
            normalized_scale_deg = scale_deg - 1
            processed_melody.append([normalized_scale_deg, acc + 1, dur])
    
    # Reshape for model input
    model_input = np.array([processed_melody])
    
    # Pad input if needed
    model_input = keras.preprocessing.sequence.pad_sequences(
        model_input, padding='post', dtype='float32'
    )
    
    # Get predictions
    predictions = model.predict(model_input)
    
    # Helper function to apply temperature
    def apply_temperature(probs, temp):
        if temp == 1.0:
            return probs
        # Apply temperature scaling
        probs = np.exp(np.log(probs) / temp)
        # Renormalize
        return probs / np.sum(probs)
    
    # Apply temperature and/or sampling
    first_scale_probs = apply_temperature(predictions[0][0], temperature)
    first_acc_probs = apply_temperature(predictions[1][0], temperature)
    second_scale_probs = apply_temperature(predictions[2][0], temperature)
    second_acc_probs = apply_temperature(predictions[3][0], temperature)
    
    if use_sampling:
        # Sample from distributions
        first_scale = np.random.choice(7, p=first_scale_probs) + 1  # Convert 0-6 to 1-7
        first_acc = np.random.choice(3, p=first_acc_probs) - 1  # Convert 0-2 to -1,0,1
        second_scale = np.random.choice(7, p=second_scale_probs) + 1
        second_acc = np.random.choice(3, p=second_acc_probs) - 1
    else:
        # Use argmax
        first_scale = np.argmax(first_scale_probs) + 1
        first_acc = np.argmax(first_acc_probs) - 1
        second_scale = np.argmax(second_scale_probs) + 1
        second_acc = np.argmax(second_acc_probs) - 1
    
    # Create actual notes/chords
    first_note = create_note_or_chord(key, first_scale, first_acc)
    second_note = create_note_or_chord(key, second_scale, second_acc)
    
    return first_note, second_note

def create_note_or_chord(key, scale_degree, accidental, create_chord=True):
    """Convert scale degree to a note or chord."""
    try:
        # Manual mapping of scale degrees to pitch names in C major
        # We'll transpose based on key later
        c_major_pitches = {
            1: 'C', 
            2: 'D', 
            3: 'E', 
            4: 'F', 
            5: 'G', 
            6: 'A', 
            7: 'B'
        }
        
        # Get base pitch name
        if scale_degree not in c_major_pitches:
            print(f"Invalid scale degree: {scale_degree}, defaulting to 1")
            scale_degree = 1
        
        base_pitch = c_major_pitches[scale_degree]
        
        # Apply accidental
        if accidental == -1:
            base_pitch += '-'  # Flat
        elif accidental == 1:
            base_pitch += '#'  # Sharp
        
        # Add octave
        pitch_with_octave = base_pitch + '4'  # Default to octave 4
        
        # Transpose to correct key
        if key.tonic.name != 'C':
            # Calculate semitone difference between C and the target key
            c_pitch = music21.pitch.Pitch('C')
            key_tonic = music21.pitch.Pitch(key.tonic.name)
            semitones = (key_tonic.ps - c_pitch.ps) % 12
            
            # Create the pitch and transpose it
            pitch_obj = music21.pitch.Pitch(pitch_with_octave)
            pitch_obj = pitch_obj.transpose(semitones)
            pitch_with_octave = pitch_obj.nameWithOctave
        
        print(f"Creating chord with root: {pitch_with_octave}")
        
        if create_chord:
            # Create chord based on pitch and scale degree
            chord_notes = [pitch_with_octave]
            
            # Add third
            third_pitch = music21.pitch.Pitch(pitch_with_octave)
            if scale_degree in [1, 4, 5] and key.mode == 'major' or \
               scale_degree in [3, 5, 6] and key.mode == 'minor':
                # Major third
                third_pitch = third_pitch.transpose(4)
            else:
                # Minor third
                third_pitch = third_pitch.transpose(3)
            chord_notes.append(third_pitch.nameWithOctave)
            
            # Add fifth
            fifth_pitch = music21.pitch.Pitch(pitch_with_octave)
            if scale_degree == 7 or (scale_degree == 2 and key.mode == 'minor'):
                # Diminished fifth
                fifth_pitch = fifth_pitch.transpose(6)
            else:
                # Perfect fifth
                fifth_pitch = fifth_pitch.transpose(7)
            chord_notes.append(fifth_pitch.nameWithOctave)
            
            return music21.chord.Chord(chord_notes)
        else:
            # Return a single note
            return music21.note.Note(pitch_with_octave)
    except Exception as e:
        print(f"Detailed error creating chord for scale degree {scale_degree}, accidental {accidental}: {str(e)}")
        # Return a C major chord as fallback
        return music21.chord.Chord(['C4', 'E4', 'G4'])

# Function to process multiple measures
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

# Create key
key = music21.key.Key('C')  # C major

# Predict harmony
predicted_chords, measures = predict_harmony_for_measures(model, test_melody, key)

# Create and save the score
score = create_score_with_harmony(test_melody, predicted_chords, measures, key)
score.write('musicxml', 'complex_melody_harmony.xml')
score.show()
