import tensorflow as tf
from tensorflow import keras
import numpy as np
import pyttsx3
import cv2

print("Preparing the model")
engine = pyttsx3.init()
'''engine.say("Preparing the model")
engine.runAndWait()'''

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Dataset loaded")

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("Pixels normalised")

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print("Encoding converted")

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
print("Architecture defined")


model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
print("The model is compiled")

print("Training the model")
'''engine.say("Training the model")
engine.runAndWait()'''

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("The model is trained")

image_path = 'Path'
print("Imaged path defined")

img = cv2.imread(image_path)
print("Imaged loaded")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_gray = cv2.resize(img_gray, (28, 28)).astype("float32") / 255

img_gray = np.reshape(img_gray, (1, 28, 28))

print("Image pre-processing completed")

prediction = np.argmax(model.predict(img_gray))

loss, accuracy = model.evaluate(x_test, y_test)
print("Test set accuracy:", accuracy)

print("TTS initialised")

print("The image is ", prediction)
engine.say("The image is " + str(prediction))
engine.runAndWait()
