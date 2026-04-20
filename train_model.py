import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

x_train=np.load("x_train.npy")
y_train=np.load("y_train.npy")

x_val=np.load("x_val.npy")
y_val=np.load("y_val.npy")

model=Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(7,activation="softmax"))

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

early_stop=EarlyStopping(
monitor="val_loss",
patience=5,
restore_best_weights=True
)

model.fit(
x_train,
y_train,
validation_data=(x_val,y_val),
epochs=30,
batch_size=64,
callbacks=[early_stop]
)

model.save("emotion_model.h5")