#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import sys
import theano
theano.config.openmp = True
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import sgd, SGD
from keras.utils import np_utils
from keras.regularizers import l2

from PIL import Image
from util import *

data_path = "/home/jiangliang/code/colorization-keras/data/"
data_file = "yuv_image/mini_class_train.npz"
cluster_center_file = "cluster_center.npz"
result_path = "/home/jiangliang/code/colorization-keras/result/"
result_file = "result.npz"
param_path = "/home/jiangliang/code/colorization-keras/params/"
param_file = "param"

print("Loading data")
data = np.load(data_path + data_file)
train_x = data["train_x"]
test_x = data["test_x"]
train_y = data["train_y"]
test_y = data["test_y"]
train_y = train_y.astype(int)
test_y = test_y.astype(int)

cluster = np.load(data_path + cluster_center_file)
cluster_center = cluster["cluster_center"]

#plt.plot(cluster_center[:, 0], cluster_center[:, 1], "*")

print(str(test_y[1][0][23748]))
#print(str(cluster_center[117, :]))
print(str(cluster_center[test_y[1][0][23748]]))

# u = train_y[:, 0, :];
# max = np.max(u)
# min = np.min(u)
#
# v = train_y[:, 1, :];
# max = np.max(v)
# min = np.min(v)

opts = {}
opts["img_patch_size"] = 7
opts["img_pixel_feature_patch_size"] = 7
opts["num_patches"] = 65536
opts["color_patch_size"] = 1
opts["batch_size"] = 128
opts["epoch"] = 10
opts["train_flag"] = True
opts["classes"] = 128

#train_x = train_x[3, :, :].reshape(1, train_x.shape[1], train_x.shape[2])
#train_y = train_y[3, :, :].reshape(1, train_y.shape[1], train_y.shape[2])

#index = train_y.reshape(train_y.shape[2])
#train_y = cluster_center[index, :]
#train_y = train_y.transpose()

#y = train_x.reshape(256, 256)
#uv = train_y.reshape(2, 256, 256)
#u = uv[0, :, :]
#v = uv[1, :, :]
#
#[r, g, b] = yuv2rgb(y, u, v)
#rm1 = np.uint8(r * 255)
#gm1 = np.uint8(g * 255)
#bm1 = np.uint8(b * 255)
#ra = np.asarray(rm1)
#ga = np.asarray(gm1)
#ba = np.asarray(bm1)
#
#converted_rgb = np.zeros((256, 256, 3), dtype = 'uint8')
#converted_rgb[:, :, 0] = ra
#converted_rgb[:, :, 1] = ga
#converted_rgb[:, :, 2] = ba
#
#im = Image.fromarray(converted_rgb)
#im.show()

pixel_model = Sequential()
pixel_model.add(Flatten(input_shape=(opts["img_pixel_feature_patch_size"] * opts["img_pixel_feature_patch_size"], )))

#pixel_model.add(Dense(opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
#pixel_model.add(Activation("softmax"))

texture_model = Sequential()
texture_model.add(Convolution2D(128, 5, 5, border_mode = 'valid', input_shape = (1, opts["img_patch_size"], opts["img_patch_size"])))
texture_model.add(Activation('relu'))
#texture_model.add(MaxPooling2D(pool_size=(2, 2)))
#texture_model.add(Convolution2D(64, 3, 3, border_mode= 'valid'))
#texture_model.add(Activation('relu'))
#texture_model.add(MaxPooling2D(pool_size=(2, 2)))
#texture_model.add(Convolution2D(128, 3, 3, border_mode= 'valid'))
#texture_model.add(Activation('relu'))
#texture_model.add(MaxPooling2D(pool_size=(2, 2)))
#texture_model.add(Activation('relu'))
#texture_model.add(MaxPooling2D(pool_size=(2, 2)))
texture_model.add(Flatten())
texture_model.add(Dense(8 * opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
texture_model.add(Activation("sigmoid"))
texture_model.add(Dense(4 * opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
texture_model.add(Activation("sigmoid"))
texture_model.add(Dense(2 * opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
texture_model.add(Activation("sigmoid"))
texture_model.add(Dense(opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
texture_model.add(Activation('softmax'))

model = Sequential()
model.add(Merge([pixel_model, texture_model], mode = "concat"))
model.add(Dense(2 * opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
model.add(Activation("sigmoid"))
model.add(Dense(2 * opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
model.add(Activation("sigmoid"))
model.add(Dense(opts["classes"], W_regularizer = l2(0.01), b_regularizer = l2(0.01)))
model.add(Activation('softmax'))

print("Compiling model")
sgd = SGD(momentum = 0.8, decay = 10e-4)
#model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
texture_model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
#pixel_model.compile(loss = 'categorical_crossentropy', optimizer = sgd)
#deal with command line parameters
if (len(sys.argv) > 1):
    if (sys.argv[1] == "train"):
        opts["train_flag"] = True
    elif (sys.argv[1] == "test"):
        opts["train_flag"] = False
    else:
        print("Wrong parameter")
        sys.exit

if (opts["train_flag"]):
    print("Get random patches")
    [train_x_patches, train_x_pixel_patches, train_y_patches] = rand_patches(train_x, train_y, opts)

    # train_y_patches[:, 0, :] = train_y_patches[:, 0, :] * 2.294
    # train_y_patches[:, 1, :] = train_y_patches[:, 1, :] * 1.626
    # train_y_patches[:, 0, :] = 0.5 * (train_y_patches[:, 0, :] + 1)
    # train_y_patches[:, 1, :] = 0.5 * (train_y_patches[:, 1, :] + 1)
    
    train_x_patches = train_x_patches.reshape(train_x_patches.shape[0], 1, opts["img_patch_size"], opts["img_patch_size"])
    train_x_vector = train_x_pixel_patches.reshape(train_x_pixel_patches.shape[0], train_x_pixel_patches.size / train_x_pixel_patches.shape[0])
    train_y_vector = train_y_patches.reshape(train_y_patches.shape[0], train_y_patches.size / train_y_patches.shape[0])

    #train_y_vector = train_y_vector.reshape(train_y_patches.shape[0])
    train_y_vector = np_utils.to_categorical(train_y_vector, opts["classes"]) 

    print("Fitting")
    #model.fit([train_x_vector, train_x_patches], train_y_vector, batch_size=opts["batch_size"], nb_epoch=opts["epoch"], show_accuracy=True, verbose=1)
    texture_model.fit([train_x_patches], train_y_vector, batch_size=opts["batch_size"], nb_epoch=opts["epoch"], show_accuracy=True, verbose=1)
    #pixel_model.fit(train_x_vector, train_y_vector, batch_size=opts["batch_size"], nb_epoch=opts["epoch"], show_accuracy=True, verbose=1)
    texture_model.save_weights(param_path + param_file, overwrite=True)
else:
    print("Load Weights")
    texture_model.load_weights(param_path + param_file)

test_x = train_x[0, :, :].reshape(1, train_x.shape[1], train_x.shape[2])
test_y = train_y[0, :, :].reshape(1, train_y.shape[1], train_y.shape[2])
test_x = test_x[0, :, :].reshape(1, test_x.shape[1], test_x.shape[2])
test_y = test_y[0, :, :].reshape(1, test_y.shape[1], test_y.shape[2])

print("Splitting test data")
[test_x_patches, test_x_pixel_patches, test_y_patches] = split_test_data(test_x, test_y, opts)
test_x_patches = test_x_patches.reshape(test_x_patches.shape[0], test_x_patches.shape[1], 1, opts["img_patch_size"], opts["img_patch_size"])
test_x_vector = test_x_pixel_patches.reshape(test_x_pixel_patches.shape[0], test_x_pixel_patches.shape[1], test_x_pixel_patches.shape[2] * test_x_pixel_patches.shape[3])
test_y_vector = test_y_patches.reshape((test_y_patches.shape[0], test_y_patches.shape[1], test_y_patches.size / test_y_patches.shape[0] / test_y_patches.shape[1]))
original_image = np.zeros((test_x_vector.shape[0], 3, np.sqrt(test_x_vector.shape[1]), np.sqrt(test_x_vector.shape[1])))
result_image  = np.zeros((test_x_vector.shape[0], 3, np.sqrt(test_x_vector.shape[1]), np.sqrt(test_x_vector.shape[1])))

print("Evaluating")
for i in range(test_x_vector.shape[0]):
    x_patch = test_x_patches[i, :, :, :]
    x_vector = test_x_vector[i, :, :]
    y_vector = test_y_vector[i, :, :]
    y_vector = y_vector.reshape(y_vector.shape[0])
    y_vector = y_vector.astype(int)
    original_color = cluster_center[y_vector, :]
    y_vector = np_utils.to_categorical(y_vector, opts["classes"])

    #[score, acc] = model.evaluate([x_vector, x_patch], y_vector, show_accuracy=True, verbose = 1)
    [score, acc] = texture_model.evaluate([x_patch], y_vector, show_accuracy=True, verbose = 1)
    print("score: " + str(score) + ", acc: " + str(acc))
    #predict_color = model.predict([x_vector, x_patch], verbose = 1)
    predict_color = texture_model.predict([x_patch], verbose = 1)
    predict_color = predict_color.argmax(1)
    predict_color = cluster_center[predict_color, :]
    predict_color = predict_color.transpose().reshape(2, np.sqrt(predict_color.shape[0]), np.sqrt(predict_color.shape[0]))

    print(str(np.sqrt(x_vector.shape[0])))
    im_size = int(np.sqrt(x_vector.shape[0]))
    print("image size: " + str(im_size))
    original_yuv = np.zeros((3, im_size, im_size))
    result_yuv = np.zeros((3, im_size, im_size))
    print(str(original_yuv.shape))
    y = x_vector[:, (x_vector.shape[1] - 1) / 2].reshape(im_size, im_size)
    original_yuv[0, :, :] = y
    result_yuv[0, :, :] = y
    original_yuv[1 : 3, :, :] = original_color.transpose().reshape(2, im_size, im_size)
    original_image[i, :, :, :] = original_yuv
    result_yuv[1 : 3, :, :] = predict_color
    # result_yuv[1, :, :] = 2 * result_yuv[1, :, :] - 1
    # result_yuv[2, :, :] = 2 * result_yuv[2, :, :] - 1
    # result_yuv[1, :, :] = result_yuv[1, :, :] / 2.294
    # result_yuv[2, :, :] = result_yuv[2, :, :] / 1.626
    result_image[i, :, :, :] = result_yuv

    for i in range(7):
        for j in range(7):
            print("origin_u: " + str(original_yuv[1, i * 25, j * 25]) + ", pred_u: " + str(result_yuv[1, i * 25, j * 25]))
            print("origin_v: " + str(original_yuv[2, i * 25, j * 25]) + ", pred_v: " + str(result_yuv[2, i * 25, j * 25]))


np.savez(result_path + result_file, original_images = original_image, result_images = result_image)
