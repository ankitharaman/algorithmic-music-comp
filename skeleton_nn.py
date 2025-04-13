from tensorflow import keras
from tensorflow.keras import layers
import music21
import numpy as np

def prepare_data(score, key=None):
    # Determine key if not provided
    if key is None:
        key = score.analyze('key')
    
    
    # Find melody and harmony parts
    melody_part = score.parts[0]
    harmony_part = score.parts[1]
    melody_measures = []
    harmony_measures = []

    # Process measures
    for m_idx, (m_melody, m_harmony) in enumerate(zip(
            melody_part.getElementsByClass('Measure'),
            harmony_part.getElementsByClass('Measure'))):
        
        # Process melody notes 
        melody_measure = []
        for note in m_melody.notesAndRests:
            if note.isRest:
                melody_measure.append((None, 0, note.quarterLength))
                continue
            
            if note.isChord:
                pitch = note.sortDiatonicAscending().pitches[-1]  # Get top note
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
        


        # Harmony Processing
        harmony_measure = []
        beat_positions_to_capture = [0, 2]  # Beat 1 and beat 3 (0-indexed)

        # Get notes at specific beat positions
        notes_by_offset = {}
        for note in m_harmony.notesAndRests:
            beat_position = int(note.offset)  # This gives the beat position (0 for beat 1, 2 for beat 3)
            if beat_position in beat_positions_to_capture:
                notes_by_offset[beat_position] = note

        # Process the specifically captured notes in order
        for beat_position in beat_positions_to_capture:
            if beat_position in notes_by_offset:
                note = notes_by_offset[beat_position]
                
                if note.isRest:
                    harmony_measure.append((None, 0, 2.0))
                    continue
                
                if note.isChord:
                    pitch = note.sortDiatonicAscending().pitches[-1]  # Get top note
                else:
                    pitch = note.pitch
                
                scale_degree = key.getScaleDegreeFromPitch(pitch)
                
                accidental = 0
                if pitch.accidental:
                    if pitch.accidental.name == 'flat':
                        accidental = -1
                    elif pitch.accidental.name == 'sharp':
                        accidental = 1
                
                harmony_measure.append((scale_degree, accidental, 2.0))
            else:
                # If we didn't find a note at this position, add a placeholder
                print(f"No note found at beat {beat_position + 1}")
                harmony_measure.append((None, 0, 2.0))

            # Add measure to the full array of measures
            harmony_measures.append(harmony_measure)

    print(f"Number of melody measures: {len(melody_measures)}")
    print(f"Number of harmony measures: {len(harmony_measures)}")


    return melody_measures, harmony_measures

def prepare_training_data(melody_measures, harmony_measures):
    X = []  # Melody sequences
    
    # Outputs for each note component
    y_first_scale = []
    y_first_acc = []
    y_second_scale = []
    y_second_acc = []
    
    for i, melody_measure in enumerate(melody_measures):
        if i >= len(harmony_measures):
            break  # Avoid index errors
            
        harmony_measure = harmony_measures[i]
        
        # Process melody - handle None scale degrees (rests)
        melody_seq = []
        for scale_deg, acc, dur in melody_measure:
            if scale_deg is None:  # Rest
                melody_seq.append([-1, 1, dur])  # Use -1 for rest scale degree, 1 for neutral accidental
            else:
                # Convert scale degree to 0-6 for model
                normalized_scale_deg = scale_deg - 1
                melody_seq.append([normalized_scale_deg, acc + 1, dur])  # Shift accidental to 0,1,2
        
        # Check if we have enough harmony notes
        if len(harmony_measure) >= 2:
            first_note = harmony_measure[0]
            second_note = harmony_measure[1]
            
            # Handle potential None values (rests) in harmony
            if first_note[0] is None:
                # Default to tonic if rest
                first_scale, first_acc = 1, 0
            else:
                first_scale, first_acc, _ = first_note
                
            if second_note[0] is None:
                # Default to tonic if rest
                second_scale, second_acc = 1, 0
            else:
                second_scale, second_acc, _ = second_note
                
            X.append(melody_seq)
            y_first_scale.append(first_scale - 1 if first_scale is not None else 0)  # Convert to 0-6
            y_first_acc.append(first_acc + 1)      # Convert to 0,1,2
            y_second_scale.append(second_scale - 1 if second_scale is not None else 0)
            y_second_acc.append(second_acc + 1)
    
    return X, {
        'first_scale_degree': np.array(y_first_scale),
        'first_accidental': np.array(y_first_acc),
        'second_scale_degree': np.array(y_second_scale),
        'second_accidental': np.array(y_second_acc)
    }


# simplified model for two note prediction
def create_model(num_scale_degrees=7):
    # Input layer for melody sequence 
    input_layer = keras.Input(shape=(None, 3))  # (scale_degree, accidental, duration)
    
    # LSTM for processing melody
    x = layers.LSTM(64, return_sequences=True)(input_layer)
    x = layers.LSTM(64)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    
    # Output two harmony notes
    # Each note needs scale degree (1-7) and accidental (-1,0,1)
    
    # First note outputs
    first_scale_degree = layers.Dense(num_scale_degrees, activation='softmax', name='first_scale_degree')(x)
    first_accidental = layers.Dense(3, activation='softmax', name='first_accidental')(x)  # -1, 0, 1
    
    # Second note outputs
    second_scale_degree = layers.Dense(num_scale_degrees, activation='softmax', name='second_scale_degree')(x)
    second_accidental = layers.Dense(3, activation='softmax', name='second_accidental')(x)
    
    model = keras.Model(
        inputs=input_layer, 
        outputs=[
            first_scale_degree, first_accidental,
            second_scale_degree, second_accidental
        ]
    )
    
    # Compile with multiple outputs
    model.compile(
        optimizer='adam',
        loss={
            'first_scale_degree': 'sparse_categorical_crossentropy',
            'first_accidental': 'sparse_categorical_crossentropy',
            'second_scale_degree': 'sparse_categorical_crossentropy',
            'second_accidental': 'sparse_categorical_crossentropy'
        },
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y):
    # Pad sequences
    X_padded = keras.preprocessing.sequence.pad_sequences(
        X, padding='post', dtype='float32'
    )
    
    model = create_model()
    
    # Train the model with early stopping for better convergence
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    model.fit(
        X_padded, y, 
        epochs=30, 
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    return model

# Make predictions on a new melody
def predict_harmony(model, melody_measure, key):
    # Preprocess melody
    processed_melody = []
    for scale_deg, acc, dur in melody_measure:
        if scale_deg is None:  # Rest
            processed_melody.append([-1, 1, dur])  # Use -1 for rest scale degree, 1 for neutral accidental
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
    
    # Extract predicted values
    first_scale = np.argmax(predictions[0][0]) + 1  # Convert back to 1-7
    first_acc = np.argmax(predictions[1][0]) - 1    # Convert back to -1,0,1
    second_scale = np.argmax(predictions[2][0]) + 1
    second_acc = np.argmax(predictions[3][0]) - 1
    
    # Create actual notes/chords
    first_note = create_note_or_chord(key, first_scale, first_acc)
    second_note = create_note_or_chord(key, second_scale, second_acc)
    
    return first_note, second_note

def create_note_or_chord(key, scale_degree, accidental, create_chord=True):
    """Convert scale degree to a note or chord."""
    scale = key.getScale()
    
    # Get the pitch for the scale degree
    pitch = scale.pitchFromDegree(scale_degree)
    
    # Apply accidental
    if accidental == -1:
        pitch = pitch.transpose(-1)
    elif accidental == 1:
        pitch = pitch.transpose(1)
    
    if create_chord:
        # Determine chord type based on scale degree
        if scale_degree in [1, 4, 5]:  # Major chords (I, IV, V)
            chord = music21.chord.Chord([pitch, pitch.transpose(4), pitch.transpose(7)])
            return chord
        elif scale_degree in [2, 3, 6]:  # Minor chords (ii, iii, vi)
            chord = music21.chord.Chord([pitch, pitch.transpose(3), pitch.transpose(7)])
            return chord
        elif scale_degree == 7:  # Diminished chord (vii°)
            chord = music21.chord.Chord([pitch, pitch.transpose(3), pitch.transpose(6)])
            return chord
    else:
        # Just return the note
        return music21.note.Note(pitch)

# Load your score
score = music21.converter.parse('your_music_file.xml')

# Get key
key = score.analyze('key')

# Prepare data
melody_measures, harmony_measures = prepare_data(score, key)

# Create training data
X, y = prepare_training_data(melody_measures, harmony_measures)

# Train model
model = train_model(X, y)

# Save model
model.save('harmony_model.h5')

# Later, load model and make predictions
loaded_model = keras.models.load_model('harmony_model.h5')

# For a new melody measure
new_melody_measure = [(1, 0, 1.0), (3, 0, 1.0), (5, 0, 1.0), (8, 0, 1.0)]  # Example melody
first_chord, second_chord = predict_harmony(loaded_model, new_melody_measure, key)
