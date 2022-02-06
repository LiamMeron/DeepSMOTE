# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

print(torch.version.cuda)  # 10.1
import time

t0 = time.time()

from decoder import Decoder
from encoder import Encoder

##############################################################################
"""args for models"""

args = {}
args["dim_h"] = 64  # factor controlling size of hidden layers
args["n_channel"] = 1  # number of channels in the input data

args["n_z"] = 300  # 600     # number of dimensions in latent space.

args["sigma"] = 1.0  # variance in n_z
args["p_lambda"] = 0.01  # hyper param for weight of discriminator loss
args["lr"] = 0.0002  # learning rate for Adam optimizer .000
args["epochs"] = 1  # 50         # how many epochs to run for
args["batch_size"] = 100  # batch size for SGD
args["save"] = True  # save weights at each epoch of training if True
args["train"] = True  # train networks if True, else load networks from

args["dataset"] = "mnist"  #'fmnist' # specify which dataset to use

number_of_folds = 5


##############################################################################


def biased_get_class1(c):

    xbeg = images[labels == c]
    ybeg = labels[labels == c]

    return xbeg, ybeg


def GenerateSamples(X, y, n_to_sample, cl):

    # fitting the model
    n_neighbors = number_of_folds + 1

    # TODO n_jobs=1 means to use one job in parallel. Consider putting this to -1 to use all processors.
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
    nearest_neighbors.fit(X)
    distances, indices = nearest_neighbors.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neighbors)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[indices[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1), X_neighbor - X_base)

    # use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl] * n_to_sample


#############################################################################
np.printoptions(precision=number_of_folds, suppress=True)

training_image_dir = ".../0_trn_img.txt"
training_labels_dir = ".../0_trn_lab.txt"

ids = os.listdir(training_image_dir)
training_image_files = [os.path.join(training_image_dir, image_id) for image_id in ids]
print(training_image_files)

ids = os.listdir(training_labels_dir)
training_label_files = [os.path.join(training_labels_dir, image_id) for image_id in ids]
print(training_label_files)

# TODO parameterize
# path on the computer where the models are stored
model_path = ".../MNIST/models/crs5/"

encoder_folders = []
decoder_folders = []
for fold in range(number_of_folds):
    current_encoder_path = model_path + "/" + str(fold) + "/bst_enc.pth"
    current_decoder_path = model_path + "/" + str(fold) + "/bst_dec.pth"
    encoder_folders.append(current_encoder_path)
    decoder_folders.append(current_decoder_path)


for fold_num in range(number_of_folds):
    print(f"Starting {fold_num}...")
    training_image_file = training_image_files[fold_num]
    training_label_file = training_label_files[fold_num]
    print(training_image_file)
    print(training_label_file)
    images = np.loadtxt(training_image_file)
    labels = np.loadtxt(training_label_file)

    print("train imgs before reshape ", images.shape)
    print("train labels ", labels.shape)

    images = images.reshape(images.shape[0], 1, 28, 28)

    print("decy ", labels.shape)
    print(collections.Counter(labels))

    print("train imgs after reshape ", images.shape)

    # TODO classes should be passed in? Uhhh, this is not used ANYWHERE. Why is it hardcoded? - WC
    classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

    # Set device to CUDA if available else use CPU (slower)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder_for_current_fold = encoder_folders[fold_num]
    decoder_for_current_fold = decoder_folders[fold_num]

    encoder = Encoder(**args)
    encoder.load_state_dict(torch.load(encoder_for_current_fold), strict=False)
    encoder = encoder.to(device)

    decoder = Decoder(**args)
    decoder.load_state_dict(torch.load(decoder_for_current_fold), strict=False)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # TODO Add element_counts_per_class as parameter?
    # imbal = [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]
    element_counts_per_class = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    resx = []
    resy = []

    # TODO This seems to be related to the number of classes, should be generic?
    for i in range(1, 10):
        images_for_class_i, labels_for_class_i = biased_get_class1(i)
        print(images_for_class_i.shape)  # (500, 3, 32, 32)
        print(labels_for_class_i[0])  # (500,)

        # encode images_for_class_i to feature space
        images_for_class_i = torch.Tensor(images_for_class_i)
        images_for_class_i = images_for_class_i.to(device)
        images_for_class_i = encoder(images_for_class_i)
        print(images_for_class_i.shape)

        images_for_class_i = images_for_class_i.detach().cpu().numpy()
        number_of_samples_needed_for_balance = (
            element_counts_per_class[0] - element_counts_per_class[i]
        )
        sample_images, sample_classes = GenerateSamples(
            images_for_class_i, labels_for_class_i, number_of_samples_needed_for_balance, i
        )
        print(sample_images.shape)  # (4500, 600)
        print(len(sample_classes))  # 4500
        sample_classes = np.array(sample_classes)
        print(sample_classes.shape)  # 4500

        """to generate samples for resnet"""
        sample_images = torch.Tensor(sample_images)
        sample_images = sample_images.to(device)

        # TODO: Remember that X are image collections and Y are label collections (in same order). I am unsure what the
        #  ximg/ximn var names are intended to be, but ximg doees not appear to be used aside from the detach.cpu.numpy
        #  conversion which apapears to take the decoded images and converting them to numpy arrays(?). I am unsure about
        #  this but it seems to read logically....
        ximg = decoder(sample_images)
        ximn = ximg.detach().cpu().numpy()
        print(ximn.shape)  # (4500, 3, 32, 32)

        resx.append(ximn)
        resy.append(sample_classes)

    # TODO (RENAME): resx1 and resy1 may be resampled X and Y (images and labels) from all 10 class runs. This takes the images
    #  and repacks the collection of 3D+ arrays into an array of rows, presumably to ensure they print easily to a txt file
    resx1 = np.vstack(resx)
    # hstack stacks the labels in column order? I guess it takes a 1-D array of labels and stores them in numpy as a single
    #  row so that they can be indexed with the same index as the image array...
    resy1 = np.hstack(resy)

    print(resx1.shape)  # (34720, 3, 32, 32)
    print(resy1.shape)  # (34720,)

    resx1 = resx1.reshape(resx1.shape[0], -1)
    print(resx1.shape)  # (34720, 3072)

    dec_x1 = images.reshape(images.shape[0], -1)
    print("decx1 ", dec_x1.shape)
    combx = np.vstack(
        (resx1, dec_x1)
    )  # Combine the 3d arrays into rows to prepare for loading back to image file
    comby = np.hstack(
        (resy1, labels)
    )  # Combine the label arrays into rows to prepare for loading back to label file

    print(combx.shape)  # (45000, 3, 32, 32)
    print(comby.shape)  # (45000,)

    image_file_name = ".../MNIST/trn_img_f/" + str(fold_num) + "_trn_img.txt"
    np.savetxt(image_file_name, combx)

    label_file_name = ".../MNIST/trn_lab_f/" + str(fold_num) + "_trn_lab.txt"
    np.savetxt(label_file_name, comby)
    print()

t1 = time.time()
print("final time(min): {:.2f}".format((t1 - t0) / 60))
