import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("fer2013.csv")

pixels = data["pixels"].tolist()

X = []
for p in pixels:
    X.append(np.fromstring(p, dtype=int, sep=' '))

X = np.array(X)
X = X / 255.0

y = data["emotion"]
y = to_categorical(y,7)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = Sequential()

model.add(Dense(2048,activation="relu",input_shape=(2304,)))
model.add(Dense(1024,activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(256,activation="relu"))
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(7,activation="softmax"))

model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])

history = model.fit(X_train,y_train,epochs=80,batch_size=128)

loss,accuracy = model.evaluate(X_test,y_test)

print("Accuracy:",accuracy*100)