import cv2
import os
import sys

args  = sys.argv

if len(args) <= 1:
    print("No arguments provided - aborting!")
    exit(0)

path = args[1]

if not os.path.exists(path):
    print("Invalid path")
    exit(0)

bnw = path + "/bnw"
colour = path + "/colour"
og = path + "/og"

os.makedirs(bnw)
os.makedirs(colour)
os.makedirs(og)

for file in os.listdir(path):
    filepath = path + "/" + file
    if os.path.isfile(filepath):
        img = cv2.imread(path + "/" + file)
        height = img.shape[0]
        width = img.shape[1]
        cutoff = width // 2

        hc = img[:, :cutoff]
        hbnw = img[:, cutoff:]
        hc = cv2.resize(hc, [100, 100], interpolation=cv2.INTER_AREA)
        hbnw = cv2.resize(hbnw, [100, 100], interpolation=cv2.INTER_AREA)

        cv2.imwrite(bnw + "/" + file, hbnw)
        cv2.imwrite(colour + "/" + file, hc)
        os.rename(filepath, og + "/" + file)
