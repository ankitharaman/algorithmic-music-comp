from music21 import *


class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [None] * size
        self.head = 0
        self.tail = 0
        self.full = False

        self.stream = stream.Stream()
        self.scale = scale.MajorScale("C")

    def append(self, item):
        self.buffer[self.head] = item

        if self.full:
            self.tail = (self.tail + 1) % self.size

        self.head = (self.head + 1) % self.size
        self.full = self.head == self.tail

        for i in range(self.size):
            scaleDegree = (abs(i - self.head)+1)%7
            n = note.Note()
            n.pitch = self.scale.pitchFromDegree(scaleDegree)
            n.duration = duration.Duration(0.5)
            self.stream.append(n)

    def is_empty(self):
        return not self.full and (self.head == self.tail)

    def clear(self):
        self.head = 0
        self.tail = 0
        self.full = False

def generate_cb_stream(user_message, k, major=True):
    words = user_message.split(" ")

    cb = CircularBuffer(len(words))
    cb.stream.insert(0, k)
    if major:
        cb.scale = scale.MajorScale(k.tonic)
    else:
        cb.scale = scale.MinorScale(k.tonic)

    for word in words:
        cb.append(word)

    return cb.stream
