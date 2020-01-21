import numpy as np
import urllib as request
import gzip
import pickle
import os

filename = [
["training_images","train-images-idx3-ubyte.gz"],
["test_images","t10k-images-idx3-ubyte.gz"],
["training_labels","train-labels-idx1-ubyte.gz"],
["test_labels","t10k-labels-idx1-ubyte.gz"]
]

if __name__ != '__main__':
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def download_mnist(overwrite=False):
    base_url = "http://yann.lecun.com/exdb/mnist/"
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in filename:
        if not os.path.exists(os.path.join(DATA_DIR, name[1])) or overwrite:
            print("Downloading "+name[1]+"...")
            request.urlretrieve(base_url+name[1], os.path.join(DATA_DIR, name[1]))
    print("Download complete.")

def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(os.path.join(DATA_DIR, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(os.path.join(DATA_DIR, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(os.path.join(DATA_DIR, 'mnist.pkl'), 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_mnist()
    save_mnist()

def load():
    with open(os.path.join(DATA_DIR, 'mnist.pkl'),'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

if __name__ == '__main__':
    init()
