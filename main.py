from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, Dense, TimeDistributed
from tensorflow.keras.optimizers import RMSprop
import numpy as np 

model = Sequential()

# We don't want to specify the sequence length
model.add(SimpleRNN(4, input_shape = (None, 2), return_sequences=True))

# Applies Dense layer to each output
model.add(TimeDistributed(Dense(2, activation='softmax')))

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = RMSprop(lr = 1.0e-2),
    metrics = ['accuracy'] )

h = [0,1]
l = [1,0]
X = np.array(
    [ [l, l, l],
      [h, h, h],
      [l, h, l],
      [h, l, h] ]
)
y = np.array(
    [ [[0], [0], [0]],
      [[1], [0], [1]],
      [[0], [1], [1]],
      [[1], [1], [0]] ]
)

model.fit(X, y, epochs = 200)

a = np.array([h, l, h, l, h, h]).reshape(1,6,-1)
#print(a)
print(model.predict(a))
