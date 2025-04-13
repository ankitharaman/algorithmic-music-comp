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
