# import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime

data_size = 1000
train_pct = 0.8 # 80% data

train_size = int(data_size * train_pct)

# create some input between -1 and 1
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x) # shuffle data

# generate output data
# y = 0.5x + 2 + noise
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size,))

# split into test and train pairs
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# specify log directory + callbacks
logdir = "logs/train-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# define callbacks for tensorboard
tensorboard_callback = keras.callbacks.TensorBoard(
  log_dir=logdir,
  histogram_freq=50,
  write_grads=True
) # for loss

# define model
model = keras.models.Sequential([
  keras.layers.Dense(16, input_dim=1),
  keras.layers.Dense(1)
])

model.summary()

model.compile(loss='mse', optimizer=keras.optimizers.SGD())

print('Training... with default parameters')
training_history = model.fit(
  x_train, # input
  y_train, # output
  batch_size=train_size,
  verbose=0,
  epochs=100,
  validation_data=(x_test, y_test),
  callbacks=[tensorboard_callback]
)

print('Average test loss: ', np.average(training_history.history['loss']))

print(model.predict([60, 25, 2])) # expect [32.0, 14.5, 2.0]
