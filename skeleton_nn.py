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
                # pitch = note.sortDiatonicAscending().pitches[-1]  # Get top note
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
                    # pitch = note.sortDiatonicAscending().pitches[-1]  # Get top note
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
        
        # Make sure we have exactly two harmony notes (assuming beat 1 and beat 3)
        if len(harmony_measure) == 2:
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
        else:
            print(f"Skipping measure {i} - expected 2 harmony notes, got {len(harmony_measure)}")
    
    print(f"Created {len(X)} training examples")
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
    
    # Compile with multiple outputs - specify metrics for each output
    model.compile(
        optimizer='adam',
        loss={
            'first_scale_degree': 'sparse_categorical_crossentropy',
            'first_accidental': 'sparse_categorical_crossentropy',
            'second_scale_degree': 'sparse_categorical_crossentropy',
            'second_accidental': 'sparse_categorical_crossentropy'
        },
        metrics={
            'first_scale_degree': ['accuracy'],
            'first_accidental': ['accuracy'],
            'second_scale_degree': ['accuracy'],
            'second_accidental': ['accuracy']
        }
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
    try:
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
            if key.mode == 'major':
                # Major key harmony
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
                # Minor key harmony (simplified)
                if scale_degree in [1, 4]:  # Minor chords (i, iv)
                    chord = music21.chord.Chord([pitch, pitch.transpose(3), pitch.transpose(7)])
                    return chord
                elif scale_degree in [3, 5, 6]:  # Major chords (III, V, VI)
                    chord = music21.chord.Chord([pitch, pitch.transpose(4), pitch.transpose(7)])
                    return chord
                elif scale_degree in [2, 7]:  # Diminished chords (ii°, vii°)
                    chord = music21.chord.Chord([pitch, pitch.transpose(3), pitch.transpose(6)])
                    return chord
        else:
            # Just return the note
            return music21.note.Note(pitch)
    except Exception as e:
        print(f"Error creating chord: {e}")
        # Return a C major chord as fallback
        return music21.chord.Chord(['C4', 'E4', 'G4'])

def train_on_multiple_scores(score_files, epochs=30, batch_size=32):
    # Initialize empty lists to hold all processed data
    all_melody_measures = []
    all_harmony_measures = []
    
    # Process each score
    for file in score_files:
        # Load the score
        score = music21.converter.parse(file)
        
        # Analyze key
        key = score.analyze('key')
        print(f"Processing {file} in key {key}")
        
        # Process the score
        melody_measures, harmony_measures = prepare_data(score, key)
        
        # Add to master lists
        all_melody_measures.extend(melody_measures)
        all_harmony_measures.extend(harmony_measures)
    
    print(f"Total melody measures collected: {len(all_melody_measures)}")
    print(f"Total harmony measures collected: {len(all_harmony_measures)}")
    
    # Create training data
    X, y_dict = prepare_training_data(all_melody_measures, all_harmony_measures)
    
    # Split data - We need to split X and each y component separately
    from sklearn.model_selection import train_test_split
    
    # First split X into train and test
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Now split each component of y
    y_train = {}
    y_test = {}
    
    # Get the indices for train and test sets
    indices = np.arange(len(X))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split each component of y according to these indices
    for key in y_dict:
        y_train[key] = y_dict[key][train_indices]
        y_test[key] = y_dict[key][test_indices]
    
    # Create and train model
    model = create_model()
    
    # Pad sequences
    X_train_padded = keras.preprocessing.sequence.pad_sequences(
        X_train, padding='post', dtype='float32'
    )
    X_test_padded = keras.preprocessing.sequence.pad_sequences(
        X_test, padding='post', dtype='float32'
    )
    
    # # Add early stopping
    # early_stopping = keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=5, restore_best_weights=True
    # )
    
    # Train model
    history = model.fit(
        X_train_padded, y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_padded, y_test),
        # callbacks=[early_stopping]
    )
    
    # Save model
    model.save('harmony_model.h5')
    
    return model, history

import os
import music21
from tensorflow import keras
import numpy as np

def train_model_on_directory(directory_path, epochs=30, batch_size=32):
    """
    Train a harmony prediction model on all XML files in the specified directory.
    
    Args:
        directory_path: Path to the directory containing XML score files
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Trained model and training history
    """
    # Find all XML files in the directory
    score_files = []
    for file in os.listdir(directory_path):
        if file.endswith('.xml') or file.endswith('.mxl'):
            full_path = os.path.join(directory_path, file)
            score_files.append(full_path)
    
    print(f"Found {len(score_files)} XML files in {directory_path}")
    
    if len(score_files) == 0:
        print("No XML files found. Please check the directory path.")
        return None, None
    
    # Train on the found files
    return train_on_multiple_scores(score_files, epochs, batch_size)

# Example usage
model, history = train_model_on_directory('scores', epochs=40, batch_size=16)

# To plot learning curves
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot the training and validation loss and accuracy curves"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot one of the accuracies (first scale degree)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['first_scale_degree_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_first_scale_degree_accuracy'], label='Validation Accuracy')
    plt.title('First Scale Degree Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Plot the training results
if history is not None:
    plot_training_history(history)

# If your scores are in a directory called 'scores' in the current working directory
model, history = train_model_on_directory('scores')


