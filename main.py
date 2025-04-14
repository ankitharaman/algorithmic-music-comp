import preprocessing
from music21 import *


def main():
    # file = "./scores/odetojoy.mxl"
    file = "./scores/Spring.mxl"
    s = converter.parse(file)
    m, h = processed_data = preprocessing.prepare_data(s)
    print(h)


if __name__ == "__main__":
    main()
