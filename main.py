#!/usr/bin/env python

import sys
import preprocessing
from music21 import *
import pickle


def run_preprocessor():
    positiveSongs = ["./scores/canon.mxl",
                     "./scores/odetojoy.mxl",
                     "./scores/VAIANA_piano_solo_de_Disney.mxl",
                     "./scores/Spring.mxl"]

    negativeSongs = ["./scores/Sonate_No._14_Moonlight_1st_Movement.mxl",
                     "./scores/Five_Hundred_Miles_2.mxl",
                     "./scores/swan_lake.mxl",
                     "./scores/Chopin_Prelude_in_E_Minor_Opus_28_No_4.mxl"]

    processedPositiveSongs = []
    processedNegativeSongs = []

    for song in positiveSongs:
        s = converter.parse(song)
        mh_pair = preprocessing.prepare_data(s)
        processedPositiveSongs.append(mh_pair)

    for song in negativeSongs:
        s = converter.parse(song)
        mh_pair = preprocessing.prepare_data(s)
        processedNegativeSongs.append(mh_pair)
    
    positive_file = open("positive.pkl", "wb")
    negative_file = open("negative.pkl", "wb")
    pickle.dump(processedPositiveSongs, positive_file)
    pickle.dump(processedNegativeSongs, negative_file)
    positive_file.close()
    negative_file.close()


def main():
    pass

if __name__ == "__main__":
    if "--preprocess" in sys.argv:
        run_preprocessor()

    main()
