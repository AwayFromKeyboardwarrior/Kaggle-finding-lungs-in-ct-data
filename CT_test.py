import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Activation,Conv2D,Flatten,Dense,MaxPool2D,Dropout,Add,LeakyReLU,UpSampling2D
from keras.models import Model,load_model
from keras.callbacks import ReduceLROnPlateau

x_train = np.load('F:\\Kaggle\\CT\\x_train.npy')
x_val = np.load('F:\\Kaggle\\CT\\x_val.npy')
y_train = np.load('F:\\Kaggle\\CT\\y_train.npy')
y_val = np.load('F:\\Kaggle\\CT\\y_val.npy')



#==========models==========

inputs = Input(shape=(256,256,1))
net = Conv2D(32,kernel_size=3,activation='relu',padding='same')(inputs)
net = MaxPool2D(pool_size=2,padding='same')(net)
net = Conv2D(64,kernel_size=3,activation='relu',padding='same')(net)
net = MaxPool2D(pool_size=2,padding='same')(net)
net = Conv2D(128,kernel_size=3,activation='relu',padding='same')(net)
net = MaxPool2D(pool_size=2,padding='same')(net)
net = Dense(128,activation='relu')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(128,kernel_size=3,activation='sigmoid',padding='same')(net)
net = UpSampling2D(size=2)(net)
net = Conv2D(64,kernel_size=3,activation='sigmoid',padding='same')(net)
net = UpSampling2D(size=2)(net)
outputs = Conv2D(1,kernel_size=3,activation='sigmoid',padding='same')(net)

model = Model(inputs=inputs,outputs=outputs)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc','mse'])

model.summary()


#==========fitting==========
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=1,batch_size=32,callbacks=[ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=10,verbose=1,mode='auto',min_lr=1e-05)])

preds = model.predict(x_val)

fig,ax = plt.subplots(len(x_val),3,figsize=(10,100))

for i,pred in enumerate(preds):
    ax[i,0].imshow(x_val[i].squeeze(),cmap='gray')
    ax[i,1].imshow(y_val[i].squeeze(),cmap='gray')
    ax[i,2].imshow(pred.squeeze(),cmap='gray')
plt.show()