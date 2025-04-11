from tensorflow import keras
from tensorflow.keras import layers
import music21
import numpy as np

def prepare_data(score, key=None):
    # Determine key if not provided
    if key is None:
        key = score.analyze('key')
    
    melody_measures = []
    harmony_pairs = []  # Will store two notes per measure
    
    # Find melody and harmony parts
    melody_part = None
    harmony_part = None
    
    for part in score.parts:
        if part.id == 'Melody' or part.id == 'RH':
            melody_part = part
        elif part.id == 'Harmony' or part.id == 'LH':
            harmony_part = part
    
    if melody_part and harmony_part:
        # Process measures
        for m_idx, (m_melody, m_harmony) in enumerate(zip(
                melody_part.getElementsByClass('Measure'),
                harmony_part.getElementsByClass('Measure'))):
            
            # Process melody notes (skipping rests)
            melody_notes = []
            for note in m_melody.notes:
                if note.isRest:
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
                melody_notes.append((scale_degree, accidental, duration))
            
            # Extract two harmony notes from the measure: first half and second half
            first_half_notes = []
            second_half_notes = []
            
            measure_length = m_harmony.barDuration.quarterLength
            midpoint = measure_length / 2
            
            for element in m_harmony.getElementsByClass(['Chord', 'Note']):
                if element.isRest:
                    continue
                
                # Determine which half of the measure this note belongs to
                offset = element.offset
                if offset < midpoint:
                    target_list = first_half_notes
                else:
                    target_list = second_half_notes
                
                # Extract the bass note (lowest note)
                if element.isChord:
                    # For chords, take the lowest note (typically the root)
                    pitch = element.bass()
                else:
                    pitch = element.pitch
                
                scale_degree = key.getScaleDegreeFromPitch(pitch)
                
                accidental = 0
                if pitch.accidental:
                    if pitch.accidental.name == 'flat':
                        accidental = -1
                    elif pitch.accidental.name == 'sharp':
                        accidental = 1
                
                target_list.append((scale_degree, accidental))
            
            # Get the first note in each half (or default if empty)
            first_note = first_half_notes[0] if first_half_notes else (1, 0)  # Default to tonic
            second_note = second_half_notes[0] if second_half_notes else (1, 0)
            
            melody_measures.append(melody_notes)
            harmony_pairs.append((first_note, second_note))
    
    return melody_measures, harmony_pairs


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

# training data prep
def prepare_training_data(melody_measures, harmony_pairs):
    X = []  # Melody sequences
    
    # Outputs for each note component
    y_first_scale = []
    y_first_acc = []
    y_second_scale = []
    y_second_acc = []
    
    for melody, (first_note, second_note) in zip(melody_measures, harmony_pairs):
        # Process melody
        melody_seq = []
        for scale_deg, acc, dur in melody:
            # Convert scale degree to 0-6 for model
            normalized_scale_deg = scale_deg - 1
            melody_seq.append([normalized_scale_deg, acc + 1, dur])  # Shift accidental to 0,1,2
        
        # Extract note components
        first_scale, first_acc = first_note
        second_scale, second_acc = second_note
        
        X.append(melody_seq)
        y_first_scale.append(first_scale - 1)  # Convert to 0-6
        y_first_acc.append(first_acc + 1)      # Convert to 0,1,2
        y_second_scale.append(second_scale - 1)
        y_second_acc.append(second_acc + 1)
    
    return X, {
        'first_scale_degree': np.array(y_first_scale),
        'first_accidental': np.array(y_first_acc),
        'second_scale_degree': np.array(y_second_scale),
        'second_accidental': np.array(y_second_acc)
    }

# training model
def train_model(X, y):
    # Pad sequences
    X_padded = keras.preprocessing.sequence.pad_sequences(
        X, padding='post', dtype='float32'
    )
    
    model = create_model()
    
    # Train the model
    model.fit(
        X_padded, y, 
        epochs=20, 
        batch_size=32,
        validation_split=0.2
    )
    
    return model

# converting notes to chords
def note_to_chord(key, scale_degree, accidental, chord_type='triad'):
    """Convert a scale degree to a full chord."""
    # Get the key's scale
    scale = key.getScale()
    
    # Adjust scale degree and convert to pitch
    pitch_class = scale.pitchFromDegree(scale_degree)
    
    # Apply accidental if needed
    if accidental == -1:  # flat
        pitch_class = pitch_class.transpose(-1)
    elif accidental == 1:  # sharp
        pitch_class = pitch_class.transpose(1)
    
    # Create a chord based on the root note
    if chord_type == 'triad':
        # Determine major vs minor based on position in scale
        if scale_degree in [1, 4, 5]:  # I, IV, V are typically major in major keys
            chord = music21.chord.Chord([pitch_class, pitch_class.transpose(4), pitch_class.transpose(7)])
        else:  # ii, iii, vi are typically minor in major keys
            chord = music21.chord.Chord([pitch_class, pitch_class.transpose(3), pitch_class.transpose(7)])
    
    # You can add other chord types here
    
    return chord

# Train the model
melody_measures, harmony_pairs = prepare_data(score)
X, y = prepare_training_data(melody_measures, harmony_pairs)
model = train_model(X, y)

# Make predictions on a new melody
def predict_harmony(model, melody_sequence, key):
    # Preprocess melody
    processed_melody = []
    for scale_deg, acc, dur in melody_sequence:
        normalized_scale_deg = scale_deg - 1
        processed_melody.append([normalized_scale_deg, acc + 1, dur])
    
    # Reshape for model input
    model_input = np.array([processed_melody])
    
    # Get predictions
    predictions = model.predict(model_input)
    
    # Extract predicted values
    first_scale = np.argmax(predictions[0][0]) + 1  # Convert back to 1-7
    first_acc = np.argmax(predictions[1][0]) - 1    # Convert back to -1,0,1
    second_scale = np.argmax(predictions[2][0]) + 1
    second_acc = np.argmax(predictions[3][0]) - 1
    
    # Convert to chords (post-processing)
    first_chord = note_to_chord(key, first_scale, first_acc)  
    second_chord = note_to_chord(key, second_scale, second_acc)
    
    return first_chord, second_chord
