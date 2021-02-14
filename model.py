import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
y_train = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

def build_model():
    model = Sequential()
    model.add(Dense(1, input_shape=[1], activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mae')
    model.load_weights('celsius.h5')
    #model.summary()
    return model
model = build_model()

#history = model.fit(x_train, y_train, epochs=50000)
pred = float(input('Enter the number: '))
print(model.predict([pred]))
from kerasify import export_model
export_model(model, 'example.model')