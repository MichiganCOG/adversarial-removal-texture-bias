import numpy as np
from matplotlib import pyplot as plt
import mnist
import cv2
import pickle
import os

    
def make(background='self', construction='tilex8', overwrite=False):
    backgrounds = ['self', 'same_label', 'correlated', 'wrong_label']
    constructions = ['tilex8', 'tilex28', 'jumble']
    
    assert background in backgrounds, 'Invalid choice of background setting: {} (valid choices are [{}]'.format(background, ','.join(backgrounds))
    assert construction in constructions, 'Invalid choice of construction setting: {} (valid choices are [{}]'.format(construction, ','.join(constructions))
    
    fname = 'mnist_mosaic_{}_{}.npy'.format(background, construction)
    if os.path.exists(fname) and not overwrite:
        return
    
    #TODO: Implement the rest and remove this block
    if not (background == 'self' and construction == 'tilex8'):
        raise NotImplementedError('Mosaic for background \'{}\' and construction \'{}\' is not yet implemented'.format(background, construction))
        
    xtrain, ytrain, xtest, ytest = mnist.load()
    # Convert to lists of images
    xtrain = [xtrain[i,:].reshape([28,28]) for i in range(xtrain.shape[0])] 
    xtest = [xtest[i,:].reshape([28,28]) for i in range(xtest.shape[0])]
    
    Ntrain = len(xtrain)
    Ntest = len(xtest)
    
    ntiles = {'tilex8': 8, 'tilex28': 28}
    if construction in ntiles:
        n = ntiles[construction]
        xtrain_mosaic = np.zeros([Ntrain, 224, 224])
        xtest_mosaic = np.zeros([Ntest, 224, 224])
        
        for i in range(Ntrain):
            xtrain_mosaic[i,:,:] = maketile(xtrain, ytrain, i, background, n)
            xtrain_mosaic[i,:,:] = np.minimum(xtrain_mosaic[i,:,:], cv2.resize(xtrain[i], dsize=(224,224)))
        
        for i in range(Ntest):
            xtest_mosaic[i,:,:] = maketile(xtest, ytest, i, background, n)
            xtest_mosaic[i,:,:] = np.minimum(xtest_mosaic[i,:,:], cv2.resize(xtest[i], dsize=(224,224)))
            
        #xtiles = [maketile(xtrain, ytrain, idx, background, n) for idx in range(Ntrain)]
        #xmasks = [cv2.resize(xtrain[idx], dsize=(224,224)) for idx in range(Ntrain)]
        #xtrain_mosaic = np.stack([np.minimum(t,m) for t,m in zip(xtiles,xmasks)], axis=0)
        
        #xtiles = [maketile(xtest, ytest, idx, background, n) for idx in range(Ntest)]
        #xmasks = [cv2.resize(xtest[idx], dsize=(224,224)) for idx in range(Ntest)]
        #xtest_mosaic = np.stack([np.minimum(t,m) for t,m in zip(xtiles,xmasks)], axis=0)
        
        #del xtiles, xmasks, xtrain, xtest
        
    elif construction == 'jumble':
        pass #TODO
    
    # Save the results
    mnist_mosaic = {'training_images': xtrain_mosaic,
                    'training_labels': ytrain,
                    'testing_images': xtest_mosaic,
                    'testing_labels': ytest}
    with open(fname, 'wb') as f:
        np.save(f, mnist_mosaic)



def maketile(xtrain, ytrain, idx, background='self', ntile=8):
    # Get n*n indices of the appropriate type
    if background == 'self':
        inds = [idx]*(ntile*ntile)
    elif background == 'same_label':
        pass #TODO
    elif background == 'correlated':
        pass #TODO
    elif background == 'wrong_label':
        pass #TODO
    
    xlist = [xtrain[ind] for ind in inds] # List of images
    xrows = [np.concatenate(tuple(xlist[i*ntile:(i+1)*ntile]), axis=1) for i in range(ntile)] # List of concat'd rows
    xtile = np.concatenate(tuple(xrows), axis=0) # Full tiled image
    
    if xtile.shape[0] != 224:
        xtile = cv2.reshape(xtile, dsize=(224,224))
    return xtile




#def self_expandx8(x):
#    xtile = np.tile(x,[8,8])
#    xbig = cv2.resize(x, dsize=(224,224))
#    return np.minimum(xtile,xbig)
#
#def make_mnist_mos(overwrite=False):
#    if os.path.exists('mnist_mosaic.pkl') and not overwrite:
#        return
#        
#    xtrain,ytrain,xtest,ytest = mnist.load()
#    xtrain = xtrain.reshape([-1,28,28])
#    xtest = xtest.reshape([-1,28,28])
#
#    xtrain_mos = np.stack([self_expandx8(xtrain[i,:,:].squeeze()) for i in range(xtrain.shape[0])], axis=0)
#    xtest_mos = np.stack([self_expandx8(xtest[i,:,:].squeeze()) for i in range(xtest.shape[0])], axis=0)
#
#    mnist_mos = {'training_images': xtrain_mos,
#                 'training_labels': ytrain,
#                 'testing_images': xtest_mos,
#                 'testing_labels': ytest}
#    with open('mnist_mosaic.pkl', 'wb') as f:
#        pickle.dump(mnist_mos,f)
#

def load(background='self', construction='tilex8'):
    fname = 'mnist_mosaic_{}_{}.npy'.format(background, construction)
    with open(fname, 'rb') as f:
        mnist_mos = np.load(f, allow_pickle=True) #TODO: This still fails
    return mnist_mos['training_images'], mnist_mos['training_labels'], mnist_mos['testing_images'], mnist_mos['testing_labels']

if __name__ == '__main__':
    make()
