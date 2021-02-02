#!/usr/bin/env python
# coding: utf-8

## Find faces in the image.

# Import packages and libraries
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import face_recognition
from IPython.display import display
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image

# Load the jpg file into a numpy array
im=image.load_img("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/unknown_2.jpg")
display(im)

### Lets see on their indiviadual pictures:

#First learning pic--ALICE
p_1 = Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_1.jpg")
display(p_1)

#Second learning pic-- MAY
p_2 = Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_2.jpg")
display(p_2)

#Third learning pic-- BRIAN
p_3 = Image.open("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_3.jpg")
display(p_3)


#This is a system of running face recognition on a single image and drawing a box around each person that was identified.

#Load a sample pictures and learn how to recognize it
person_1 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_1.jpg")
person_1_encoding = face_recognition.face_encodings(person_1)[0]

person_2 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_2.jpg")
person_2_encoding = face_recognition.face_encodings(person_2)[0]

person_3 = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/person_3.jpg")
person_3_encoding = face_recognition.face_encodings(person_3)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    person_1_encoding,
    person_2_encoding,
    person_3_encoding
    
]
known_face_names = [
    "Alice",
    "May",
    "Brian"
]
print('Learned encoding for', len(known_face_encodings), 'images.'

#Finally, we load the image we looked at in the first cell, find the faces in the image and compare them with the encodings the library generated in the previous step. We can see that library now correctly recognizes Alice, May and Brian in the input.

# Load an image with an unknown face
from PIL import ImageFont
unknown_image = face_recognition.load_image_file("C:/Users/nisht/Anaconda3_n/envs/LL/face_recognition/Ch06/unknown_2.jpg")
fontsize = 15
font = ImageFont.truetype("arial.ttf", fontsize)

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
pil_image = Image.fromarray(unknown_image)

# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 20, bottom - text_height - 5), name, fill=(255, 255, 255, 255),font=font)

# Remove the drawing library from memory as per the Pillow docs
del draw
# Display the resulting image
display(pil_image)
