from music21 import *
import random

class Node:
    def __init__(self, value):
        self.value = value
        self.color = 'red'
        self.left = None
        self.right = None
        self.parent = None

class RBTree:
    def __init__(self):
        self.root = None

    def rotate_right(self, pivot):
        new_pivot = pivot.left
        if not new_pivot:
            return
        
        pivot.left = new_pivot.right
        if new_pivot.right:
            new_pivot.right.parent = pivot
       
        new_pivot.parent = pivot.parent
        
        if not pivot.parent:
            self.root = new_pivot
        elif pivot == pivot.parent.right:
            pivot.parent.right = new_pivot
        else:
            pivot.parent.left = new_pivot

        new_pivot.right = pivot
        pivot.parent = new_pivot

    def rotate_left(self, pivot):
        new_pivot = pivot.right
        if not new_pivot:
            return
        
        pivot.right = new_pivot.left
        if new_pivot.left:
            new_pivot.left.parent = pivot
       
        new_pivot.parent = pivot.parent
        
        if not pivot.parent:
            self.root = new_pivot
        elif pivot == pivot.parent.left:
            pivot.parent.left = new_pivot
        else:
            pivot.parent.right = new_pivot

        new_pivot.left = pivot
        pivot.parent = new_pivot

    def adjust(self, node):
        data = []
        while node != self.root and node.parent.color == 'red':
            grandparent = node.parent.parent
            # checks if the parent node is the left child
            if node.parent == grandparent.left:
                uncle = grandparent.right

                # case 1: there are 3 red nodes under a black node
                if uncle and uncle.color == 'red':
                    uncle.color = 'black'
                    node.parent.color = 'black'
                    grandparent.color = 'red'
                    data.append(("recolor", node.parent.value, uncle.value, 
                                 grandparent.value))
                    node = grandparent
                else:
                    # case 2: left rotation first and then case 3
                    if node == node.parent.right:
                        self.rotate_left(node.parent)
                        data.append(("left", node.parent.value))
                        node = node.left

                    # case 3: right rotation
                    self.rotate_right(grandparent)
                    data.append(('right', grandparent.value))
                    node.parent.color = 'black'
                    grandparent.color = 'red'

            else:
                # redoing cases for if parent node is the right child
                uncle = grandparent.left
                if uncle and uncle.color == 'red':
                    uncle.color = 'black'
                    node.parent.color = 'black'
                    grandparent.color = 'red'
                    data.append(("recolor", node.parent.value, uncle.value, 
                                 grandparent.value))
                    node = grandparent
                else:
                    # case 2: right rotation first and then case 3
                    if node == node.parent.left:
                        self.rotate_right(node.parent)
                        data.append(("right", node.parent.value))
                        node = node.right

                    # case 3: left rotation
                    self.rotate_left(grandparent)
                    data.append(('left', grandparent.value))
                    node.parent.color = 'black'
                    grandparent.color = 'red'
        
        self.root.color = 'black'
        return data

    def insert_node(self, value):
        node = Node(value)
        path = []

        if self.root is None:
            node.color = 'black'
            self.root = node
            return node, path, []
        
        curr_node = self.root

        while curr_node:
            path.append((curr_node.value, curr_node.color))
            
            if value < curr_node.value:
                if not curr_node.left:
                    curr_node.left = node
                    node.parent = curr_node
                    break
                curr_node = curr_node.left
            elif value > curr_node.value:
                if not curr_node.right:
                    curr_node.right = node
                    node.parent = curr_node
                    break
                curr_node = curr_node.right
            else:
                return node, path, []

        data = self.adjust(node)
        return node, path, data


def get_scale_pitches(key='C', major=True):
    sc = scale.MajorScale(key.tonic) if major else scale.MinorScale(key.tonic)
    low = str(key.tonic) + '4'
    high = str(key.tonic) + '5'
    lowest_pitch = pitch.Pitch(low)
    highest_pitch = pitch.Pitch(high)

    pitches = []
    curr_pitch = lowest_pitch
    while curr_pitch.midi <= highest_pitch.midi:
        if sc.getScaleDegreeFromPitch(curr_pitch) is not None:
            pitches.append(curr_pitch.midi)
        curr_pitch = pitch.Pitch(curr_pitch.midi + 1)

    return pitches

def clean_accidental(n):
    n.pitch.accidental = None
    n.pitch.accidentalDisplay = False
    return n

def notes_from_path(s, path):
    for value, color in path:
        n = note.Note(value)
        n.quarterLength = 0.5
        if color == 'red':
            n.volume = volume.Volume(velocity=90)
        else:
            n.volume = volume.Volume(velocity=50)
        # s.append(clean_accidental(n))
        s.append(n)

def insert_pitch(s, pitch):
    n = note.Note(pitch)
    n.quarterLength = 2.0
    n.articulations.append(articulations.Accent())
    n.volume = volume.Volume(velocity=150)
    # s.append(clean_accidental(n))
    s.append(n)

def rotation(s, mid_pitch, direction='left'):
    n1 = note.Note(mid_pitch)
    n1.quarterLength = 1.0
    n1.volume = volume.Volume(velocity=40 if direction == 'left' else 120)
    # s.append(clean_accidental(n1))
    s.append(n1)
    
    n2 = note.Note(mid_pitch)
    n2.quarterLength = 1.0
    n2.volume = volume.Volume(velocity=120 if direction == 'left' else 40)
    # s.append(clean_accidental(n2))
    s.append(n2)

def recolor(s, parent, uncle, grandparent):
    for pitch, vel1, vel2 in [
        (parent, 90, 50),
        (uncle, 90, 50),
        (grandparent, 50, 90)
    ]:
        n1 = note.Note(pitch)
        n1.volume = volume.Volume(velocity=vel1)
        # s.append(clean_accidental(n1))
        s.append(n1)

        n2 = note.Note(pitch)
        n2.volume = volume.Volume(velocity=vel2)
        # s.append(clean_accidental(n2))
        s.append(n2)

def play_adjustments(s, adjustments):
    for i in range(len(adjustments)):
        if adjustments[i][0] == 'recolor':
            pitch1 = adjustments[i][1]
            pitch2 = adjustments[i][2]
            pitch3 = adjustments[i][3]
            recolor(s, pitch1, pitch2, pitch3)
        elif adjustments[i][0] == 'right':
            pitch = adjustments[i][1]
            rotation(s, pitch, 'right')
        else:
            pitch = adjustments[i][1]
            rotation(s, pitch, 'left')
    return s

def generate_rb_stream(user_message, k, major=True):
    scale_pitches = get_scale_pitches(key=k, major=major)
    random.shuffle(scale_pitches)
    insert_order = scale_pitches

    rb_tree = RBTree()
    s = stream.Stream()
    s.insert(0, k)

    for i in range(len(insert_order)):
        node, path, adjustment = rb_tree.insert_node(insert_order[i])
        notes_from_path(s, path)
        insert_pitch(s, node.value)
        play_adjustments(s, adjustment)

    return s
