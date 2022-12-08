import os
import sys

import cv2


SIZE = 64


args  = sys.argv

if len(args) <= 1:
    print("No path provided - aborting!")
    exit(0)

path = args[1]

if not os.path.exists(path):
    print("Invalid path")
    exit(0)

bnw = os.path.join(path, "black_and_white")
colour = os.path.join(path, "colour")

os.makedirs(bnw)
os.makedirs(colour)

for file in os.listdir(path):
    filepath = path + "/" + file
    if os.path.isfile(filepath):
        img = cv2.imread(path + "/" + file)
        height = img.shape[0]
        width = img.shape[1]
        cutoff = width // 2

        hc = img[:, :cutoff]
        hbnw = img[:, cutoff:]
        hc = cv2.resize(hc, [SIZE, SIZE], interpolation=cv2.INTER_AREA)
        hbnw = cv2.resize(hbnw, [SIZE, SIZE], interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(bnw, file), hbnw)
        cv2.imwrite(os.path.join(colour, file), hc)
        os.remove(filepath)
