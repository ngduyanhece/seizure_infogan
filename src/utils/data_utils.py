import os
# from keras.utils import np_utils
import numpy as np
# import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from sklearn.utils import shuffle

SEIZURE_DATA_PATH = "/Users/dapxichlo/Desktop/My_own/seizure_data"


def load_seizure():
    ictal_files = [file_i for file_i in os.listdir(SEIZURE_DATA_PATH) if "1_ictal" in file_i][0:500]
    interictal_files = [file_i for file_i in os.listdir(SEIZURE_DATA_PATH) if "1_interictal" in file_i][0:500]
    ictal_array = []
    interictal_array = []
    for ictal_file in ictal_files:
        ictal_file_path = os.path.join(SEIZURE_DATA_PATH,ictal_file)
        data = np.load(ictal_file_path)
        if data.shape[-1] == 22:
            ictal_array.append(data)
    for interictal_file in interictal_files:
        interictal_file_path = os.path.join(SEIZURE_DATA_PATH,interictal_file)
        data = np.load(interictal_file_path)
        if data.shape[-1] == 22:
            interictal_array.append(data)

    ictal_array = np.array(ictal_array)
    interictal_array = np.array(interictal_array)
    ictal_labels = np.zeros(ictal_array.shape[0])
    interictal_labels = np.ones(interictal_array.shape[0])
    X = np.concatenate((ictal_array,interictal_array))
    y = np.concatenate((ictal_labels,interictal_labels))
    X,y = shuffle(X,y)
    X_train = X[0:-100]
    y_train = y[0:-100]
    X_test = X[-100:]
    y_test = y[-100:]
    return X_train,y_train,X_test,y_test


def gen_batch(X, batch_size):
    while True:
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        yield X[idx]

def sample_noise(noise_scale, batch_size, noise_dim):

    return np.random.normal(scale=noise_scale, size=(batch_size, noise_dim[0]))


def sample_cat(batch_size, cat_dim):

    y = np.zeros((batch_size, cat_dim[0]), dtype="float32")
    random_y = np.random.randint(0, cat_dim[0], size=batch_size)
    y[np.arange(batch_size), random_y] = 1
    return y

def get_disc_batch(X_real_batch, Y_real_batch, generator_model, batch_size, cat_dim, noise_dim, noise_scale=0.5,type="real",label_smoothing=False):

    # Create X_disc: alternatively only generated or real images
    if type == "fake":
        # Pass noise to the generator
        # y_cat = sample_cat(batch_size, cat_dim)
        # get some labels of the X_real_batch
        # batch_slice = int(batch_size/2)
        # y_cat[0:batch_slice,:] = Y_real_batch[0:batch_slice,:]
        noise_input = sample_noise(noise_scale, batch_size, noise_dim)
        # Produce an output
        X_disc = generator_model.predict([Y_real_batch, noise_input],batch_size=batch_size)
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        return X_disc, y_disc, noise_input
    else:
        # batch_slice = int(batch_size / 2)
        X_disc = X_real_batch
        y_disc = np.zeros((X_disc.shape[0], 1), dtype=np.uint8)
        # y_cat = sample_cat(batch_size, cat_dim)
        # y_cat[0:batch_slice, :] = Y_real_batch[0:batch_slice, :]
        if label_smoothing:
            y_disc[:, 0] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            y_disc[:, 0] = 1
        return X_disc, y_disc

    # return X_disc, y_disc, y_cat

def get_gen_batch(batch_size, cat_dim, noise_dim, noise_scale=0.5):

    X_gen = sample_noise(noise_scale, batch_size, noise_dim)
    y_gen = np.zeros((X_gen.shape[0], 1), dtype=np.uint8)
    # y_gen[:, 0] = 1
    # batch_slice = int(batch_size / 2)
    # y_cat = sample_cat(batch_size, cat_dim)
    # y_cat[0:batch_slice, :] = Y_real_batch[0:batch_slice, :]
    # return X_gen, y_gen, y_cat
    return X_gen, y_gen

def accuracy(p_y,y_ind):
    labels = np.argmax(y_ind,axis=1)
    p_labels = np.argmax(p_y,axis=1)
    return np.mean(p_labels == labels)