#!/usr/bin/env python
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from util import rand_patches, yuv2rgb

data_path = "/home/jiangliang/code/colorization-keras/data/"
data_file = "yuv_image/train.npz"
result_file = "yuv_image/class_train.npz"
mini_result_file = "yuv_image/mini_class_train.npz"
middle_result_file = "yuv_image/middle_class_train.npz"
cluster_center_file = "cluster_center.npz"

opts = {}
opts["img_patch_size"] = 35
opts["img_pixel_feature_patch_size"] = 3
opts["num_patches"] = 30000
opts["color_patch_size"] = 11
opts["batch_size"] = 128
opts["epoch"] = 1
opts["train_flag"] = False

print("Loading data")
data = np.load(data_path + data_file)

train_x = data["train_x"]
test_x = data["test_x"]
train_y = data["train_y"]
test_y = data["test_y"]


if (opts["train_flag"]):
    cluster = KMeans(n_clusters = 128, max_iter = 3000, n_init = 10, precompute_distances = True, n_jobs = -1, verbose = 1, copy_x = True)
    print("Getting random patches")
    [temp1, temp2, y] = rand_patches(train_x, train_y, opts)
    print(str(y.shape))
    y_vector = np.zeros((y.shape[0] * y.shape[2], 2))
    print(str(y_vector.shape))
    for i in range(y.shape[0]):
        temp = y[i, :, :]
        temp = temp.transpose()
        y_vector[i * y.shape[2] : (i + 1) * y.shape[2], :] = temp
    print("fitting model")
    cluster.fit(y_vector, y=None)
    print("saving data")
    np.savez(data_path + cluster_center_file, cluster_center = cluster.cluster_centers_)
else:
    cluster = KMeans(n_clusters = 128, max_iter = 3000, n_init = 10, precompute_distances = True, n_jobs = -1, verbose = 1, copy_x = True)
    #cluster = KMeans(n_clusters = 128)
    #cluster = KMeans(n_clusters = 128, max_iter = 3000, n_init = 1, precompute_distances = True, n_jobs = -1, verbose = 1, copy_x = True)
    param = np.load(data_path + cluster_center_file)
    cluster_center = param["cluster_center"]
    cluster.cluster_centers_ = cluster_center

    train_y_vector = np.zeros((train_y.shape[0] * train_y.shape[2], 2))
    train_labels = np.zeros((train_y.shape[0], 1, train_y.shape[2]))
    for i in range(train_y.shape[0]):
        print("i: " + str(i))
        temp = train_y[i, :, :]
        temp = temp.transpose()
        #train_y_vector[i * train_y.shape[2] : (i + 1) * train_y.shape[2], :] = temp
        label = cluster.predict(temp)
        train_labels[i, :, :] = label.reshape(1, label.size)
    #train_labels = cluster.predict(train_y_vector)
    #train_labels = train_labels.reshape(train_y.shape[0], 1, train_y.shape[2])
  
    temp = (train_labels == -1)  
    if (temp.any()):
        print("Wrong")
    else:
        print("Right")

    test_y_vector = np.zeros((test_y.shape[0] * test_y.shape[2], 2))
    test_labels = np.zeros((test_y.shape[0], 1, test_y.shape[2]))
    test_labels = test_labels - 1
    for i in range(test_y.shape[0]):
        temp = test_y[i, :, :]
        temp = temp.transpose()
        #test_y_vector[i * test_y.shape[2] : (i + 1) * test_y.shape[2], :] = temp
        label = cluster.predict(temp)
        test_labels[i, :, :] = label.reshape(1, label.size)
    #test_labels= cluster.predict(test_y_vector)
    #test_labels = test_labels.reshape(test_y.shape[0], 1, test_y.shape[2])
    temp = (test_labels == -1)  
    if (temp.any()):
        print("Wrong")
    else:
        print("Right")
    
    train_index = np.arange(train_y.shape[0])
    test_index = np.arange(test_y.shape[0])
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)
    
    mini_train_index = train_index[0 : 100]
    mini_test_index = test_index[0 : 30]
    middle_train_index = train_index[0 : 2000]
    middle_test_index = test_index[0 : 500]
    
    mini_train_x = train_x[mini_train_index, :, :]
    mini_test_x = test_x[mini_test_index, :, :]
    mini_train_labels = train_labels[mini_train_index, :, :]
    mini_test_labels = test_labels[mini_test_index, :, :]
    
    middle_train_x = train_x[middle_train_index, :, :]
    middle_test_x = test_x[middle_test_index, :, :]
    middle_train_labels = train_labels[middle_train_index, :, :]
    middle_test_labels = test_labels[middle_test_index, :, :]

    np.savez(data_path + result_file, train_x = train_x, test_x = test_x, train_y = train_labels, test_y = test_labels)
    np.savez(data_path + mini_result_file, train_x = mini_train_x, test_x = mini_test_x, train_y = mini_train_labels, test_y = mini_test_labels)
    np.savez(data_path + middle_result_file, train_x = middle_train_x, test_x = middle_test_x, train_y = middle_train_labels, test_y = middle_test_labels)
print("finish")


