import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path="mnist.npz")
print(X_train.shape, X_test.shape)
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print(type(Y_train[0]))
print(X_train[0])
print(X_train.max())

# Normalize
X_train = X_train / 255
X_test = X_test / 255

# Model building - Config 1
model1 = Sequential()
model1.add(Dense(10, input_dim=784, activation='softmax'))
model1.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
cp = ModelCheckpoint('Mymodel.keras', monitor='val_loss', save_best_only=True, mode='min')
print(model1.summary())
res = model1.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[cp])
a = model1.evaluate(X_test, Y_test)

# Plot data
plt.plot(res.history['val_loss'], label='validation loss')
plt.plot(res.history['loss'], label='training loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()