import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

data = pd.read_csv("fer2013.csv")

pixels = data["pixels"].tolist()

faces = []

for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split()]
    face = np.asarray(face).reshape(48,48)
    faces.append(face)

faces = np.asarray(faces)

faces = faces.astype("float32")

faces = faces / 255.0

faces = np.expand_dims(faces,-1)

emotions = data["emotion"].values

emotions = to_categorical(emotions,7)

x_train = faces[data["Usage"]=="Training"]
y_train = emotions[data["Usage"]=="Training"]

x_val = faces[data["Usage"]=="PublicTest"]
y_val = emotions[data["Usage"]=="PublicTest"]

x_test = faces[data["Usage"]=="PrivateTest"]
y_test = emotions[data["Usage"]=="PrivateTest"]

np.save("x_train.npy",x_train)
np.save("y_train.npy",y_train)

np.save("x_val.npy",x_val)
np.save("y_val.npy",y_val)

np.save("x_test.npy",x_test)
np.save("y_test.npy",y_test)

print("Preprocessing Completed")