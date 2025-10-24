import  numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras

df = pd.read_csv('insurance_data.csv')
print(df.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[['age', 'affordability']], df.bought_insurance, test_size=0.2, random_state=42)

# print(len(x_train))
# print(len(x_test))
# print(len(y_train))
# print(len(y_test))

# x_train_scaled = x_train.copy()
# x_test_scaled = x_test.copy()

# x_train_scaled['age'] = x_train_scaled['age'] / 100
# x_test_scaled['age'] = x_test_scaled['age'] / 100

# print(x_train_scaled.head())

# model = keras.Sequential([
#     keras.layers.Dense(1, input_shape=(2,), 
#     activation='sigmoid', 
#     kernel_initializer='ones', 
#     bias_initializer='zeros')
# ])

# model.compile(optimizer='adam', 
#     loss='binary_crossentropy', 
#     metrics=['accuracy'])

# model.fit(x_train_scaled, y_train, epochs=1000)

# model.evaluate(x_test_scaled, y_test)
# model.save('insurance_model.h5')

# y_pred = model.predict(x_test_scaled)
# print(y_pred)
# print(x_test_scaled)

insurance_model = keras.models.load_model('insurance_model.h5')
print(insurance_model.predict(np.array([[26, 1]])))
print(insurance_model.predict(np.array([[19, 1]])))
print(insurance_model.predict(np.array([[26, 0]])))
print(insurance_model.predict(np.array([[188, 0]])))

