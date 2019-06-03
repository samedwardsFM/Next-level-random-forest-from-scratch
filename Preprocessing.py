import h5py
import numpy as np


def load_data():
    #path = '/Users/minoh/Documents/Usyd2018-2/COMP5318 Machine Learning/Assignment1/data/' #change path
    #path = 'C:/SE-software/Data Science/COMP5318/Assignment 1/'
    #path = 'C:/Users/Owner/Documents/Data Science/COMP5418/'
    path = 'input/'

    with h5py.File(path + 'images_training.h5','r') as H:
      data = np.copy(H['data'])
    with h5py.File(path + 'labels_training.h5','r') as H:
      label = np.copy(H['label'])
    with h5py.File(path + 'images_testing.h5','r') as H:
      test = np.copy(H['data'])
    with h5py.File(path + 'labels_testing_2000.h5','r') as H:
      test_label = np.copy(H['label'])

    test_labels = np.zeros((3000))
    test_labels = np.concatenate((test_label, test_labels), axis=0)

    flat_data = data.reshape(30000, 784)
    label_data = label.reshape(30000, 1)
    flat_test = test.reshape(5000, 784)
    label_test = test_labels.reshape(5000,1)

    combined_data = np.insert(flat_data, [784], label_data, axis=1)
    combined_test = np.insert(flat_test, [784], label_test, axis=1)
    return combined_data, combined_test


# Split dataset into k-fold train/validation sets

def kfold(dataset, n_splits=6):
  # shuffle the data
  np.random.shuffle(dataset)
  n = len(dataset) // n_splits
  X_train = []
  y_train = []
  X_val = []
  y_val = []

  for i in range(1, n_splits + 1):
    if i < n_splits:
      X_val.append(dataset[n * (i - 1):n * i][:, :784])
      y_val.append(dataset[n * (i - 1):n * i][:, 784:])
    else:
      X_val.append(dataset[n * (i - 1):][:, :784])
      y_val.append(dataset[n * (i - 1):][:, 784:])

  for i in range(1, n_splits + 1):
    a = dataset[0:n * (i - 1)]
    b = dataset[n * i:]
    X_train.append(np.concatenate((a, b), axis=0)[:, :784])
    y_train.append(np.concatenate((a, b), axis=0)[:, 784:])

  return X_train, y_train, X_val, y_val


# Split dataset into train/validation sets

def split_data(dataset, row):
  # shuffle the data
  np.random.shuffle(dataset)
  X_train = []
  y_train = []
  X_val = []
  y_val = []
  X_train.append(dataset[:row,:784])
  y_train.append(dataset[:row,784:])
  X_val.append(dataset[row:,:784])
  y_val.append(dataset[row:,784:])
  return X_train, y_train, X_val, y_val

def zerocenter(X_train, X_val):
  X_train_centered = [x_t - np.mean(x_t, axis=0) for x_t in X_train]
  X_val_centered = [x_v - np.mean(x_t, axis=0) for x_t, x_v in zip(X_train,X_val)]

  return X_train_centered, X_val_centered


def PCA(X, p):
  # X is dataset with m samples and n features, p is a float number between 0 and 1.
  # Reduce data of matrix X from n dimensions to k dimensions. Returns low dimensional representation Z.

  # Get data dimension
  (m, n) = X.shape  # m samples, n features

  # Compute covariance matrix
  cov = (1.0 / (m - 1)) * X.T.dot(X)  # X-transposed * X gives an (n x n) matrix

  # Perform Singular Value Decomposition to compute eigenvectors of the covariance matrix cov
  U, S, V = np.linalg.svd(cov, full_matrices=True)  # U is (n x n) matrix

  # Diagonal matrix S contains singular values of X. With these singular values, we can compute the proportion of retained variance over total variance.
  total = np.sum(S)
  k = 1
  partial = np.sum(S[:k + 1])
  proportion = partial / total
  while proportion < p:
    k += 1
    partial = np.sum(S[:k + 1])
    proportion = partial / total

  # Select first k columns of U
  U_reduced = U[:, : k]

  Z = X.dot(U_reduced)  # low dimensional representation of data

  return Z, U_reduced, k

def PCA_transform(X_train_centered, X_val_centered, p):
    # Apply PCA to 6 training sets
    Z_train = []
    U_reduced = []
    Z_val = []
    K = []
    for x_tc in X_train_centered:
        z_train, u_reduced, k = PCA(x_tc,p)
        Z_train.append(z_train)
        U_reduced.append(u_reduced)
        K.append(k)
    print(k,'Principal components retained')
    for x_vc, u_r in zip(X_val_centered, U_reduced):
        Z_val.append(x_vc.dot(u_r))

    return Z_train, Z_val, U_reduced, K

def preprocess_k_fold(training_set,p):
    """
    produce k-fold training and validation sets
    :param training_set:
    :param p:
    :return:
    """
    X_train_K, Y_train_K, X_val_K, Y_val_K = kfold(training_set)
    X_train_K, X_val_K = zerocenter(X_train_K, X_val_K)
    X_train_K, X_val_K, U_reduced, K = PCA_transform(X_train_K, X_val_K, p)
    return X_train_K, Y_train_K, X_val_K, Y_val_K

def preprocess_split(training_set,p):
    """
    split into taining and validation sets
    :param training_set:
    :param p:
    :return:
    """
    X_train_25K, Y_train_25K, X_val, Y_val = split_data(training_set, 25000)
    X_train_25K, X_val = zerocenter(X_train_25K, X_val)
    X_train_25K, X_val, U_reduced, K = PCA_transform(X_train_25K, X_val, p)
    return X_train_25K, Y_train_25K, X_val, Y_val

def preprocess_all(training_set, test_set,p):
    """
    final preprocessing, no split
    :param training_set:
    :param test_set:
    :param p:
    :return:
    """
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_train.append(training_set[:, :784])
    Y_train.append(training_set[:, 784:])
    X_test.append(test_set[:, :784])
    Y_test.append(test_set[:, 784:])
    if p > -1:
        X_train, X_test = zerocenter(X_train, X_test)
        X_train, X_test, U_reduced, K = PCA_transform(X_train, X_test, p)
    return X_train, Y_train, X_test, Y_test


