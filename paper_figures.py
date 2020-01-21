import numpy as np
import matplotlib.pyplot as plt
import mosaic_mnist

# Figure 1: Examples of the different configurations of Mosaic MNIST
def figure1(show=True):
    # Consistent MosNIST
    idx = 0 #TODO: Choose a pretty one
    images,labels = mosaic_mnist.load('filename') #TODO: figure out where this will be
    img = images[idx,:,:].squeeze()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    
    # Inconsistent MosNIST
    idx = 0 #TODO: Choose a pretty one
    images,labels = mosaic_mnist.load('filename') #TODO: figure out where this will be
    img = images[idx,:,:].squeeze()
    plt.subplot(1,3,2)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    
    # Malicious MosNIST
    idx = 0 #TODO: Choose a pretty one
    images,labels = mosaic_mnist.load('filename') #TODO: figure out where this will be
    img = images[idx,:,:].squeeze()
    plt.subplot(1,3,3)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
