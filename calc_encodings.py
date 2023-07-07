# import the necessary packages
import utils
import face_recognition
import argparse
import pickle
import cv2
import os
import logging


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset, then initialize
# out data list (which we'll soon populate)
print("[INFO] quantifying faces...")
imagePaths = list(utils.list_images(args["dataset"]))
data = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths[:10]):
	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	print(imagePath, end=" ")
	image = cv2.imread(imagePath)
	print(image.shape, end=" ")
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	print("found boxes =", len(boxes), end=" ")
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
	# choose one face
	d = [{"imagePath": imagePath,
          "loc": boxes[0] if len(boxes) else boxes, 
          "encoding": encodings[0] if len(encodings) else encodings}]
	data.extend(d)
	print("data length =", len(data))
	
# dump the facial encodings data to disk
print("[INFO] serializing encodings...")
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()