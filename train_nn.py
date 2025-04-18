#!/usr/bin/env python

from tensorflow import keras
# from tensorflow.keras import layers
import music21
import numpy as np
import sys
import os

layers = keras.layers

def prepare_data(score, key=None):
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

def create_model(num_scale_degrees=7):
    input_layer = keras.Input(shape=(None, 3))
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(input_layer)
    x = layers.Dropout(0.3)(x)  # Add dropout to prevent overfitting
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.3)(x)
    
    # Deeper dense layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
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

def train_on_multiple_scores(score_files, output_file, epochs=30, batch_size=32):
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
    
    # Train model
    history = model.fit(
        X_train_padded, y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_padded, y_test),
    )
    
    # Save model
    model.save(output_file)
    
    return model, history

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

if __name__ == "__main__":
    if "--pos" in sys.argv:
        positive_model, history = train_model_on_directory('./scores/positive_scores', "positive_harmony_model.h5")
    elif "--neg" in sys.argv:
        negative_model, history = train_model_on_directory('./scores/negative_scores', "negative_harmony_model.h5")
    else:
        positive_model, history = train_model_on_directory('./scores/positive_scores', "positive_harmony_model.h5")
        negative_model, history = train_model_on_directory('./scores/negative_scores', "negative_harmony_model.h5")
