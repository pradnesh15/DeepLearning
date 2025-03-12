import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

#generate synthetic data
np.random.seed(42)
X=np.random.rand(100,1)
Y=(2*X+np.random.randn(100,1))*2

#define model
model=Sequential()

#add layers
model.add(Dense(1,input_dim=1))

#compile
model.compile(optimizer=SGD(learning_rate=0.01),loss='mean_squared_error')

#train
model.fit(X,Y,epochs=100)

#predictions
output=model.predict(X)

#plot
plt.scatter(X,Y,label='original data')
plt.plot(X,output,label='predicted data')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()