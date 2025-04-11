from music21 import *

positiveSongs = ["./canon.mxl", "./odetojoy.mxl",
                 "./VAIANA_piano_solo_de_Disney.mxl"]
negativeSongs = ["./Sonate_No._14_Moonlight_1st_Movement.mxl",
                 "./Five_Hundred_Miles_2.mxl"]

for song in positiveSongs:
    s = converter.parse(song)
    s.show()

# b.show('midi')
