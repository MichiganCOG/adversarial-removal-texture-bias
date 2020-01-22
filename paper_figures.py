import numpy as np
import matplotlib.pyplot as plt
import mosaic_mnist as mm
import mosaic_mnist.mnist as mnist

# Figure 1: Examples of the different configurations of Mosaic MNIST
def figure1(show=True):
    idx = 626
    
    # Original MNIST
    x1,y1,x2,y2 = mnist.load()
    img = x2[idx,:].reshape([28,28])
    plt.subplot(1,4,1)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)

    # Consistent MosNIST
    images,labels = mm.load('consistent_test')
    img = images[idx,:,:].squeeze()
    plt.subplot(1,4,2)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    
    # Inconsistent MosNIST
    images,labels = mm.load('consistent_test')  # Change this to inconsistent once files are ready
    img = images[idx,:,:].squeeze()
    plt.subplot(1,4,3)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    
    # Malicious MosNIST
    images,labels = mm.load('malicious_test')
    img = images[idx,:,:].squeeze()
    plt.subplot(1,4,4)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    
    if show:
        plt.show()




if __name__ == '__main__':
    figure1()
