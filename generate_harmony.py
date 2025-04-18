import music21
import numpy as np
from redblack import generate_rb_stream
from circularbuffer import generate_cb_stream
from hashtable import generate_hashtable_stream
from tensorflow import keras

model = None

def prepare_melody_data(score, key):
    # Find melody and harmony parts
    melody_part = score.makeMeasures()

    melody_measures = []

    # Process measures
    for m_idx, m_melody in enumerate(melody_part.getElementsByClass('Measure')):

        # Process melody notes 
        melody_measure = []
        for note in m_melody.notesAndRests:
            if note.isRest:
                melody_measure.append((None, 0, note.quarterLength))
                continue

            if note.isChord:
                pitch = note.root()
            else:
                pitch = note.pitch

            scale_degree = key.getScaleDegreeFromPitch(pitch)

            accidental = 0
            if pitch.accidental:
                if pitch.accidental.name == 'flat':
                    accidental = -1
                elif pitch.accidental.name == 'sharp':
                    accidental = 1

            duration = note.quarterLength
            melody_measure.append((scale_degree, accidental, duration))

        # Add measure to the full array of measures
        melody_measures.append(melody_measure)

    return melody_measures



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
        model_input, padding='post', dtype='float32')

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
        # # Manual mapping of scale degrees to pitch names in C major
        # # We'll transpose based on key later
        # c_major_pitches = {
        #     1: 'C', 
        #     2: 'D', 
        #     3: 'E', 
        #     4: 'F', 
        #     5: 'G', 
        #     6: 'A', 
        #     7: 'B'
        # }
        #
        # # Get base pitch name
        # if scale_degree not in c_major_pitches:
        #     print(f"Invalid scale degree: {scale_degree}, defaulting to 1")
        #     scale_degree = 1
        #
        # base_pitch = c_major_pitches[scale_degree]

        pitches = key.getScale().getPitches()
        base_pitch = str(pitches[scale_degree-1])

        # Apply accidental
        if accidental == -1:
            base_pitch += '-'  # Flat
        elif accidental == 1:
            base_pitch += '#'  # Sharp

        # Add octave
        # pitch_with_octave = base_pitch + '4'  # Default to octave 4

        print(f"Creating chord with root: {base_pitch}")

        if create_chord:
            # Create chord based on pitch and scale degree
            chord_notes = [base_pitch]

            # Add third
            third_pitch = music21.pitch.Pitch(base_pitch)
            if scale_degree in [1, 4, 5] and key.mode == 'major' or \
               scale_degree in [3, 5, 6] and key.mode == 'minor':
                # Major third
                third_pitch = third_pitch.transpose(4)
            else:
                # Minor third
                third_pitch = third_pitch.transpose(3)
            chord_notes.append(third_pitch.nameWithOctave)

            # Add fifth
            fifth_pitch = music21.pitch.Pitch(base_pitch)
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
            return music21.note.Note(base_pitch)
    except Exception as e:
        print(f"Detailed error creating chord for scale degree {scale_degree}, accidental {accidental}: {str(e)}")
        # Return a C major chord as fallback
        return music21.chord.Chord(['C4', 'E4', 'G4'])

# Function to process multiple measures
def predict_harmony_for_measures(model, measures, key):
    # Process each measure
    predicted_chords = []
    for measure in measures:
        first_chord, second_chord = predict_harmony(model, measure, key)
        predicted_chords.append((first_chord, second_chord))

    return predicted_chords, measures

# Create a score with the results
def create_score_with_harmony(predicted_chords, measures, key):
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
            # print(scale_deg, acc, dur)
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


if __name__ == "__main__":
    # Load the trained model
    model = keras.models.load_model('negative_harmony_model.h5')

    # Create key
    k = "Eb"
    key = music21.key.Key(k)  # C major
    user_message = "This is a test message to generate a melody."

    circular_buffer_stream = generate_cb_stream(user_message, key, major=True)
    hashtable_stream = generate_hashtable_stream(user_message, key, major=True)
    redblack_stream = generate_rb_stream(user_message, key, major=True)

    # melody_measures = prepare_melody_data(hashtable_stream, None)
    cb_melody = prepare_melody_data(circular_buffer_stream, key)
    ht_melody = prepare_melody_data(hashtable_stream, key)
    rb_melody = prepare_melody_data(redblack_stream, key)

    combined_melody = ht_melody + cb_melody + rb_melody

    # Predict harmony
    cb_predicted_chords, cb_measures = predict_harmony_for_measures(model, cb_melody, key)
    ht_predicted_chords, ht_measures = predict_harmony_for_measures(model, ht_melody, key)
    rb_predicted_chords, rb_measures = predict_harmony_for_measures(model, rb_melody, key)

    combined_predicted_chords = ht_predicted_chords + cb_predicted_chords + rb_predicted_chords
    combined_measures = ht_measures + cb_measures + rb_measures

    # Create and save the score
    score = create_score_with_harmony(combined_predicted_chords, combined_measures, key)
    score.write('musicxml', 'complex_melody_harmony.xml')
