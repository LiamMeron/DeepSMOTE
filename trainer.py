import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset

import mnist_loader
from decoder import Decoder
from encoder import Encoder

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG,
                    filename=f'data/logs/trainer.log', filemode='a+')
log = logging.getLogger()
log.debug("DEBUG IS ENABLED")
##############################################################################
"""set models, loss functions"""


# control which parameters are frozen / free for optimization
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


##############################################################################
"""functions to create SMOTE images"""


def biased_get_class(decoded_images, decoded_labels, c):
    xbeg = decoded_images[decoded_labels == c]
    ybeg = decoded_images[decoded_labels == c]

    return xbeg, ybeg


def Gen_Samples(X, y, n_to_sample, cl):
    # determining the number of samples to generate
    # n_to_sample = 10

    # fitting the model
    n_neigh = 5 + 1
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nearest_neighbors.fit(X)
    distances, indices = nearest_neighbors.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    image_base = X[base_indices]
    image_neighbors = X[indices[base_indices, neighbor_indices]]

    samples = image_base + np.multiply(
        np.random.rand(n_to_sample, 1), image_neighbors - image_base
    )

    # use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl] * n_to_sample


def train(
        training_img_dir: Path = (".../MNIST/trn_img/"),
        training_label_dir: Path = Path(".../MNIST/trn_lab/"),
        model_output_path: Path = Path(".../MNIST/models/crs5/"),
):
    print(torch.version.cuda)  # 10.1
    t3 = time.time()
    ##############################################################################
    """args for AE"""

    args = get_args()

    ###############################################################################

    # NOTE: Download the training ('.../0_trn_img.txt') and label files
    # ('.../0_trn_lab.txt').  Place the files in directories (e.g., ../MNIST/trn_img/
    # and /MNIST/trn_lab/).  Originally, when the code was written, it was for 5 fold
    # cross validation and hence there were 5 files in each of the
    # directories.  Here, for illustration, we use only 1 training and 1 label
    # file (e.g., '.../0_trn_img.txt' and '.../0_trn_lab.txt').
    log.info(f"Loading training images from {training_img_dir}")
    image_training_files = [f.absolute() for f in training_img_dir.iterdir()]
    log.info(f"Loaded files {image_training_files}")

    log.info(f"Loading training labels from {training_img_dir}")
    label_training_files = [f.absolute() for f in training_label_dir.iterdir()]
    log.info(f"Loaded files {image_training_files}")

    # For each batch of training files (each fold)
    for fold_number in range(len(label_training_files)):
        log.info(f"Training on image file: {image_training_files[fold_number]}")
        log.info(f"Training on label file: {label_training_files[fold_number]}")

        image_dataset = np.load(
            str(image_training_files[fold_number]), allow_pickle=True
        )
        labels_dataset = np.load(
            str(label_training_files[fold_number]), allow_pickle=True
        )
        run_training_fold(
            args, fold_number, image_dataset, labels_dataset, model_output_path
        )

    t4 = time.time()
    print(f"Total runtime: {(t4 - t3) / 60:.2f} minutes")


def run_training_fold(
        args, training_fold, decoded_images, decoded_labels, model_output_path
):
    log.info(f"Starting training fold: {training_fold}")
    log.debug(f"Creating Encoder")
    encoder = Encoder(**args)
    log.debug(f"Creating Decoder")
    decoder = Decoder(**args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Training on {device.upper()}")
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)

    # Loading in images

    log.info(f"Training images before reshape: {decoded_images.shape}")
    log.info(f"Training labels: {decoded_labels.shape}")
    # log.info(f"Label counts: {collections.Counter(decoded_labels)}")
    decoded_images = decoded_images.reshape(decoded_images.shape[0], 1, 28, 28)
    log.info(f"Training images after reshape: {decoded_images.shape}")
    batch_size = args["batch_size"]
    num_workers = 0
    # torch.Tensor returns float so if want long then use torch.tensor
    tensor_images = torch.Tensor(decoded_images)
    tensor_labels = torch.tensor(decoded_labels, dtype=torch.long)
    mnist_bal = TensorDataset(tensor_images, tensor_labels)
    train_loader = torch.utils.data.DataLoader(
        mnist_bal, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    best_loss = np.inf
    t0 = time.time()
    log.info(f"Fold {training_fold} started at {t0}")
    if args["train"]:
        learning_rate = args["lr"]
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

        for epoch in range(args["epochs"]):
            log.info(f"Training epoch: {epoch}")
            train_epoch(
                best_loss,
                criterion,
                dec_optim,
                decoded_images,
                decoded_labels,
                decoder,
                device,
                enc_optim,
                encoder,
                epoch,
                training_fold,
                model_output_path,
                train_loader,
            )

        # in addition, store the final model (may not be the best) for
        # informational purposes
        path_enc = model_output_path / f"fold_{training_fold}" / "final_encoder.pth"
        path_dec = model_output_path / f"fold_{training_fold}" / "final_decoder.pth"
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
        print()
    t1 = time.time()
    log.info(f"Fold {training_fold} took: {(t1 - t0) / 60:.2f} minutes")


def train_epoch(
        best_loss,
        criterion,
        dec_optim,
        decoded_images,
        decoded_labels,
        decoder,
        device,
        enc_optim,
        encoder,
        epoch,
        i,
        model_output_path,
        train_loader,
):
    train_loss = 0.0
    tmse_loss = 0.0
    tdiscr_loss = 0.0
    # train for one epoch -- set nets to train mode
    encoder.train()
    decoder.train()
    for images, labels in train_loader:
        # zero gradients for each batch
        encoder.zero_grad()
        decoder.zero_grad()
        # print(images)
        images, labels = images.to(device), labels.to(device)
        labsn = labels.detach().cpu().numpy()

        # run images
        z_hat = encoder(images)

        x_hat = decoder(z_hat)  # decoder outputs tanh
        # print('xhat ', x_hat.size())
        # print(x_hat)
        mse = criterion(x_hat, images)
        # print('mse ',mse)

        resx = []
        resy = []

        # Get a random class
        current_training_class = np.random.choice(10, 1)
        images_in_current_training_class = decoded_images[
            decoded_labels == current_training_class
            ]
        labels_in_current_training_class = decoded_labels[
            decoded_labels == current_training_class
            ]
        number_of_images_in_training_class = len(images_in_current_training_class)
        number_of_samples = min(number_of_images_in_training_class, 100)
        indices_of_samples = np.random.choice(
            list(range(len(images_in_current_training_class))),
            number_of_samples,
            replace=False,
        )
        image_samples = images_in_current_training_class[indices_of_samples]
        label_samples = labels_in_current_training_class[indices_of_samples]

        number_of_epoch_training_samples = len(image_samples)
        xcminus = np.arange(1, number_of_epoch_training_samples)

        xcplus = np.append(xcminus, 0)
        xcnew = image_samples[[xcplus], :]
        xcnew = xcnew.reshape(
            xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4]
        )

        xcnew = torch.Tensor(xcnew)
        xcnew = xcnew.to(device)

        # encode image_samples to feature space
        image_samples = torch.Tensor(image_samples)
        image_samples = image_samples.to(device)
        image_samples = encoder(image_samples)

        image_samples = image_samples.detach().cpu().numpy()

        xc_enc = image_samples[[xcplus], :]
        xc_enc = np.squeeze(xc_enc)

        xc_enc = torch.Tensor(xc_enc)
        xc_enc = xc_enc.to(device)

        ximg = decoder(xc_enc)

        mse2 = criterion(ximg, xcnew)

        comb_loss = mse2 + mse
        comb_loss.backward()

        enc_optim.step()
        dec_optim.step()

        train_loss += comb_loss.item() * images.size(0)
        tmse_loss += mse.item() * images.size(0)
        tdiscr_loss += mse2.item() * images.size(0)
        log.info(
            "Train Loss: {:.6f}\tmse loss: {:.6f} \tdiscr loss: {:.6f}".format(
                epoch, train_loss, tmse_loss, tdiscr_loss
            )
        )

    # print avg training statistics
    train_loss = train_loss / len(train_loader)
    tmse_loss = tmse_loss / len(train_loader)
    tdiscr_loss = tdiscr_loss / len(train_loader)
    log.info(
        "Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}".format(
            epoch, train_loss, tmse_loss, tdiscr_loss
        )
    )
    # store the best encoder and decoder models
    # here, /crs5 is a reference to 5 way cross validation, but is not
    # necessary for illustration purposes
    if train_loss < best_loss:
        log.info(
            f"Epoch({epoch}) training loss = {train_loss}\tBest training loss = {best_loss}"
        )
        log.info(f"Saving epoch({epoch}) as best model for fold({i})")
        path_enc = model_output_path / f"fold_{i}" / "best_encoder.pth"
        path_dec = model_output_path / f"fold_{i}" / "best_decoder.pth"

        for path in [path_enc.parent, path_dec.parent]:
            if not path.exists():
                os.makedirs(path)

        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)

        best_loss = train_loss


def get_args():
    args = {}
    args["dim_h"] = 64  # factor controlling size of hidden layers
    args["n_channel"] = 1  # 3    # number of channels in the input data
    args["n_z"] = 300  # 600     # number of dimensions in latent space.
    args["sigma"] = 1.0  # variance in n_z
    args["p_lambda"] = 0.01  # hyper param for weight of discriminator loss
    args["lr"] = 0.0002  # learning rate for Adam optimizer .000
    args["epochs"] = 200  # how many epochs to run for
    args["batch_size"] = 100  # batch size for SGD
    args["save"] = True  # save weights at each epoch of training if True
    args["train"] = True  # train networks if True, else load networks from
    args["dataset"] = "mnist"  # 'fmnist' # specify which dataset to use
    return args


if __name__ == "__main__":
    log.info("Starting DeepSMOTE Training")
    x_train, t_train, x_test, t_test = mnist_loader.load()
    run_training_fold(get_args(), 1, x_train, t_train,
                      model_output_path=Path(f"./data/out/equipment_failures/{datetime.now().date()}/models"))
    #         f"./data/out/equipment_failures/{datetime.now().date()}/models"
    #     ),)
    # train(
    #     training_img_dir=Path(
    #         r"D:\willc\Documents\School\Fall 2021\CMSC451-Capstone\repos\CS-22-324-Predictive-Modeling-Anticipating-Equipment-Failure\data\equipment_failure\images"
    #     ),
    #     training_label_dir=Path(
    #         r"D:\willc\Documents\School\Fall 2021\CMSC451-Capstone\repos\CS-22-324-Predictive-Modeling-Anticipating-Equipment-Failure\data\equipment_failure\labels"
    #     ),
    #     model_output_path=Path(
    #         f"./data/out/equipment_failures/{datetime.now().date()}/models"
    #     ),
    # )
