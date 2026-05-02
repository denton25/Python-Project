import numpy as np
from scipy.io import loadmat

def load_emnist_mat(path):
    data = loadmat(path)

    train = data["dataset"]["train"][0, 0]
    test = data["dataset"]["test"][0, 0]

    # FIX: unwrap labels properly
    y_train_raw = train["labels"][0, 0].astype(np.int32).flatten()
    y_test_raw = test["labels"][0, 0].astype(np.int32).flatten()

    # FIX: EMNIST Balanced labels are 1–47 → convert to 0–46
    y_train = y_train_raw - 1
    y_test = y_test_raw - 1

    # Load images
    x_train = train["images"][0, 0].reshape((-1, 28, 28)).astype(np.float32)
    x_test = test["images"][0, 0].reshape((-1, 28, 28)).astype(np.float32)

    # Fix EMNIST rotation
    x_train = np.transpose(x_train, (0, 2, 1))[:, ::-1, :]
    x_test = np.transpose(x_test, (0, 2, 1))[:, ::-1, :]

    # Normalize
    x_train /= 255.0
    x_test /= 255.0

    return x_train, y_train, x_test, y_test






