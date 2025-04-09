from music21 import *

class MusicalHashTable:
    def __init__(self, size=7):
        """Initialize a hash table with a given size."""
        self.size = size
        self.table = [None] * size
        self.count = 0  
        self.melody = []

    def _hash(self, value):
        hash_val = hash(value)
        return hash_val % self.size

    def insert(self, value):
        """Insert a key-value pair into the hash table using the value for hashing."""
        if self.count >= self.size * 0.7:  # Resize if load factor exceeds 0.7
            self._resize()

        index = self._hash(value)  # Hash based on value instead of key
        original_index = index

        # Linear probing to find an empty slot or the key if it exists
        # Add all viewed indexes into a sub-array
        beat = []
        beat.append(index)
        while self.table[index] is not None:
            # Linear probing
            index = (index + 1) % self.size
            beat.append(index)

            # Do nothing if table is full
            if index == original_index:
                return

        # Insert in the found slot
        self.table[index] = value
        self.count += 1

        # Insert the index into the melody array
        self.melody.append(beat)

    def _resize(self):
        """Resize the hash table when it's getting full."""
        print("Resize!")
        old_table = self.table
        self.size = self.size * 2
        self.table = [None] * self.size
        self.count = 0

        # Reinsert all existing elements
        for item in old_table:
            if item is not None:
                value = item
                self.insert(value)

    def __str__(self):
        """Return a string representation of the hash table."""
        result = "{"
        for item in self.table:
            if item is not None:
                result += f"{item}, "
            else:
                result += "_, "
        if result != "{":
            result = result[:-2]  # Remove trailing comma and space
        result += "}"
        return result

    def __len__(self):
        """Return the number of elements in the hash table."""
        return self.count
    

def array_to_music21(array_data, key="C", scale_type="major"):
    # Create scale
    if scale_type.lower() == "major":
        sc = scale.MajorScale(key)
    elif scale_type.lower() == "minor":
        sc = scale.MinorScale(key)
    else:
        raise ValueError(f"Unsupported scale type: {scale_type}")
    
    # Create stream
    s = stream.Stream()
    
    # Add a time signature (4/4 by default)
    s.append(meter.TimeSignature('4/4'))
    
    # Process each beat (subarray)
    for beat in array_data:
        # Determine duration based on number of elements
        if len(beat) == 1:
            # Quarter note
            dur_value = 1.0
        else:
            # For other durations: 1.0 (quarter) divided by number of elements
            dur_value = 1.0 / len(beat)
        
        # Add each note in the beat
        for scale_degree in beat:
            # Calculate octave and actual scale degree
            octave = (4 + (scale_degree // 7) % 2)
            actual_degree = scale_degree % 7
            
            # Get the pitch from the scale
            pitch = sc.pitchFromDegree(actual_degree + 1)  # +1 because scale degrees start at 1
            
            # Create a note with the appropriate duration
            n = note.Note(pitch)
            n.octave = octave
            
            # Set duration
            if len(beat) == 3:
                # Special case for triplets
                n.duration = duration.Duration(dur_value)
                n.duration.quarterLength = 1/3
            else:
                n.duration = duration.Duration(dur_value)
            
            # Add the note to the stream
            s.append(n)
    
    return s



ht = MusicalHashTable()

# Insert key-value pairs
ht.insert("I")
print(ht)
ht.insert("went")
print(ht)
ht.insert("to")
print(ht)
ht.insert("the")
print(ht)
ht.insert("store")
print(ht)
ht.insert("again")
print(ht)
ht.insert("I")
print(ht)
ht.insert("I")
print(ht)
ht.insert("went")
print(ht)
ht.insert("to")
print(ht)
ht.insert("the")
print(ht)
ht.insert("store")
print(ht)
ht.insert("again")
print(ht)
ht.insert("I")

# Print the hash table
print(ht)
print(ht.melody)


score = array_to_music21(ht.melody, key="D", scale_type="minor")
score.show()
