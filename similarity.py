
# coding: utf-8

# In[4]:


import random
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.datasets import cifar10
from keras.utils import to_categorical
import cv2
import numpy as np

# read data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[2]:


# limit the amount of the data
# train data
ind_train = random.sample(list(range(x_train.shape[0])), 2000)
x_train = x_train[ind_train]
y_train = y_train[ind_train]

def resize_data(data):
    data_upscaled = np.zeros((data.shape[0], 140, 140, 3))
    for i, img in enumerate(data):
        large_img = cv2.resize(img, dsize=(140, 140), interpolation=cv2.INTER_CUBIC)
        data_upscaled[i] = large_img

    return data_upscaled

# resize train and  test data
x_train_resized = resize_data(x_train)
x_test_resized = resize_data(x_test)

# make explained variable hot-encoded
y_train_hot_encoded = to_categorical(y_train)
y_test_hot_encoded = to_categorical(y_test)


# In[5]:


inc_model = InceptionV3(weights='imagenet', include_top=False)

# get layers and add average pooling layer
x = inc_model.output
x = GlobalAveragePooling2D()(x)

# add fully-connected layer
x = Dense(512, activation='relu')(x)

# add output layer
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inc_model.input, outputs=predictions)

# freeze pre-trained model area's layer
for layer in inc_model.layers:
    layer.trainable = False

# update the weight that are added
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x_train_resized, y_train_hot_encoded)

# choose the layers which are updated by training
layer_num = len(model.layers)
for layer in model.layers[:279]:
    layer.trainable = False

for layer in model.layers[279:]:
    layer.trainable = True

# training
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_resized, y_train_hot_encoded, batch_size=128, epochs=5, shuffle=True,  validation_split=0.3)


# In[ ]:


def predict_scores(img):
    resized_img = resize_data(img)
    return history.model.predict(resized_img)

print(predict_scores(x_test[:5]))


# In[ ]:


ind_test = random.sample(list(range(x_train.shape[0])), 2000)
x_test = x_test[ind_test]


# In[6]:


predicted = predict_scores(x_test)


# In[7]:


from scipy.spatial.distance import cdist

def find_similar_image(target, predicted, imgs, n):
    score = predict_scores(np.array([target]))
    return [target, imgs[np.argsort(cdist(score, predicted))[0][:n]]]

similar_imgs = find_similar_image(x_test[11], predicted, x_test, 10)


# In[8]:


def make_plot(similar_imgs):
    plt.figure(figsize=(32,32))
    plt.subplot(1, 11, 1)
    plt.imshow(similar_imgs[0])
    for i, img in enumerate(similar_imgs[1]):
        plt.subplot(1, 11, i+2)
        plt.imshow(img)
    plt.show()
make_plot(similar_imgs

