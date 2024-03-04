import keras
from keras.optimizers import SGD, adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.utils import np_utils
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report, confusion_matrix


source = 'C:/Users/srinivasulu.buddi/.spyder-py3/Project/all_samples'
destination = 'C:/Users/srinivasulu.buddi/.spyder-py3/Project/new_samples'

img_width, img_height = 224, 224

listing = os.listdir(source)
num_samples =size(listing)
print('Total number of samples = ',num_samples)

for file in listing:
	im = Image.open(source + '\\' + file)
	img = im.resize((img_width, img_height))
	img.save(destination + '\\' + file, "png")

imlist = os.listdir(destination)


immatrix = array([array(Image.open(destination + '\\' + im2)).flatten()
	for im2 in imlist], 'f')

label = np.ones((num_samples,), dtype = int)
label[0:192] = 0
label[192:234] = 1
label[234:346] = 2



data, Label = shuffle(immatrix, label, random_state=4)
train_data = [data, Label]


(X,y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 3)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = to_categorical(y_train, 3)
Y_test = to_categorical(y_test, 3)


# AlexNet model
model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11,11),
                 strides=(4,4), padding='valid',
                 input_shape=(img_width, img_height,3),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), 
                       strides=(2,2), 
                       padding='valid'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


history=model.fit(X_train, Y_train, batch_size=32, 
                  epochs=50,
                  verbose=1,
                  validation_data=(X_test, Y_test))


train_acc = model.evaluate(X_train, Y_train, verbose=0)
print('\nTrain Accuracy: %.3f' %train_acc[1])

model.save('alexnet_save.h5')


# Check Performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']



# loss
plt.plot(loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(acc, label='train acc')
plt.plot(val_acc, label='val acc')
plt.xlabel('Epochs')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')






# predict probabilities for test set

y_pred = model.predict(X_test, verbose=0)
y_pred=np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
print('confusion matrix \n',cm)
print('\n')
print('classification Report')
target_names = ['asleep', 'awake', 'drowsy']
print(classification_report(y_test,y_pred,target_names=target_names))


score=model.evaluate(X_test,Y_test,verbose=0)
recall = recall_score(y_test, y_pred , average = "macro")
precision = precision_score(y_test, y_pred , average="macro")
f1 = f1_score(y_test, y_pred, average="macro")


print('Test Accuracy = %.3f\n' %score[1])
print("Precision = %.3f\n" %precision)
print("Recall = %.3f\n" %recall)
print("f1score = %.3f\n" %f1)
