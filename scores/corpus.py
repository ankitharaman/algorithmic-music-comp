from music21 import *

positiveSongs = ["./canon.mxl", "./odetojoy.mxl",
                 "./VAIANA_piano_solo_de_Disney.mxl", "./Spring.mxl"]

negativeSongs = ["./Sonate_No._14_Moonlight_1st_Movement.mxl",
                 "./Five_Hundred_Miles_2.mxl"]

for song in positiveSongs:
    s = converter.parse(song)
    keyStr = str(s.analyze('key'))

    scl = None
    if "minor" in keyStr:
        scl = scale.MinorScale(keyStr.split(" ")[0])
        print(scl)
    else:
        scl = scale.MajorScale(keyStr.split(" ")[0])
        print(scl)

    for note in s:
        print(note)
