import numpy as np
from matplotlib import pyplot as plt
import mnist
import cv2
import pickle
import os
import random
import multiprocessing as mp

DATA_DIR = 'mnist_data'

    
def make(background='self', construction='tilex8', overwrite=False, num_workers=-1):
    backgrounds = ['self', 'same_label', 'correlated', 'wrong_label']
    constructions = ['tilex8', 'tilex28', 'jumble']
    
    # Input validation
    assert background in backgrounds, 'Invalid choice of background setting: {} (valid choices are [{}]'.format(background, ','.join(backgrounds))
    assert construction in constructions, 'Invalid choice of construction setting: {} (valid choices are [{}]'.format(construction, ','.join(constructions))
    assert num_workers >= -1, 'Invalid number of parallel workers: {}'.format(num_workers)
    
    # Check for output file
    fname = 'mnist_mosaic_{}_{}.npy'.format(background, construction)
    fname = os.path.join(DATA_DIR, fname)
    if os.path.exists(fname) and not overwrite:
        return
        
    # Load data
    print('Loading MNIST...')
    xtrain, ytrain, xtest, ytest = mnist.load()
    
    # Convert to lists of images
    xtrain = [xtrain[i,:].reshape([28,28]) for i in range(xtrain.shape[0])] 
    xtest = [xtest[i,:].reshape([28,28]) for i in range(xtest.shape[0])]
    
    Ntrain = len(xtrain)
    Ntest = len(xtest)
    
    # Initialize results
    print('Initializing results...')
    xtrain_mosaic = np.zeros([Ntrain, 224, 224], dtype=np.uint8)
    xtest_mosaic = np.zeros([Ntest, 224, 224], dtype=np.uint8)
    
    # Define callback functions for parallelization
    def callback_train(results):
        xtrain_mosaic[results[0]:results[1],:,:] = results[2]
    def callback_test(results):
        xtest_mosaic[results[0]:results[1],:,:] = results[2]
    
    ### Construct mosaic dataset ###
    # Tiled construction
    ntiles = {'tilex8': 8, 'tilex28': 28}
    if construction in ntiles:
        n = ntiles[construction]
        
        # No parallelization
        if num_workers == 1 or num_workers == 0:
            print('Making training set...')
            for i in range(Ntrain):
                xtrain_mosaic[i,:,:] = make_tile_image(xtrain, ytrain, i, background, n)
            print('Making test set...')
            for i in range(Ntest):
                xtest_mosaic[i,:,:] = make_tile_image(xtest, ytest, i, background, n)
                
        # Data parallelization
        else:
            if num_workers == -1:
                num_workers == mp.cpu_count()
            print('Opening pool of {} workers...'.format(num_workers))
            pool = mp.Pool(num_workers)
            
            # Start workers for training set
            for worker_id in range(num_workers):
                pool.apply_async(wrap_parallel,
                                 args=(make_tile_image, worker_id, num_workers),
                                 kwds={'dataset': xtrain,
                                       'labels': ytrain,
                                       'background': background,
                                       'ntile': n},
                                 callback=callback_train)
            pool.close()
            pool.join()
            
            # Start workers for test set
            pool = mp.Pool(num_workers)
            for worker_id in range(num_workers):
                pool.apply_async(wrap_parallel,
                                 args=(make_tile_image, worker_id, num_workers),
                                 kwds={'dataset': xtest,
                                       'labels': ytest,
                                       'backgroud': background,
                                       'ntile': n},
                                 callback=callback_test)
            pool.close()
            pool.join()
            
            
    # Jumble construction    
    elif construction == 'jumble':
        
        # No paralellization
        if num_workers == 1 or num_workers == 0:
            print('Making training set...')
            for i in range(Ntrain):
                xtrain_mosaic[i,:,:] = make_jumbled_image(xtrain, ytrain, i, background)
            print('Making test set...')
            for i in range(Ntest):
                xtest_mosaic[i,:,:] = make_jumbled_image(xtest, ytest, i, background)
        
        # Data parallelization:
        else:
            if num_workers == -1:
                num_workers = mp.cpu_count()
            print('Opening pool of {} workers...'.format(num_workers))
            pool = mp.Pool(num_workers)
            
            # Start workers for training set
            for worker_id in range(num_workers):
                pool.apply_async(wrap_parallel,
                                 args=(make_jumbled_image, worker_id, num_workers),
                                 kwds={'dataset': xtrain,
                                       'labels': ytrain,
                                       'background': background},
                                 callback=callback_train)
            pool.close()
            pool.join()
            
            # Start workers for test set
            pool = mp.Pool(num_workers)
            for worker_id in range(num_workers):
                pool.apply_async(wrap_parallel,
                                 args=(make_jumbled_image, worker_id, num_workers),
                                 kwds={'dataset': xtest,
                                       'labels': ytest,
                                       'background': background},
                                 callback=callback_test)
            pool.close()
            pool.join()
        
    
    '''
    ### Visualize the dataset ###
    for i in range(25):
        plt.subplot(5,10,2*i+1)
        plt.imshow(xtrain_mosaic[i,:,:].squeeze(), cmap='gray')
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.subplot(5,10,2*i+2)
        plt.imshow(xtrain[i], cmap='gray')
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
    plt.show()
    '''
    ### Save the results ###
    print('Saving results...')
    mnist_mosaic = {'training_images': xtrain_mosaic,
                    'training_labels': ytrain,
                    'testing_images': xtest_mosaic,
                    'testing_labels': ytest}
    with open(fname, 'wb') as f:
        np.save(f, mnist_mosaic)


# Choose N indices of digits to make up the composite digit based on the background type
def select_indices(labels, idx, background, N):
    # Use the composite digit N times
    if background == 'self':
        inds = [idx]*N
    # Use N digits of the same label as composite digit
    elif background == 'same_label':
        probs = [1. if y == labels[idx] else 0. for y in labels]
        probs = [p / sum(probs) for p in probs]
        inds = np.random.choice(len(labels), (N,), False, probs)
    # Choose digits randomly, favoring same-labeled digits highly
    elif background == 'correlated':
        probs = [1. if y == labels[idx] else 0.05 for y in labels]       #TODO: tune this
        probs = [p / sum(probs) for p in probs]
        inds = np.random.choice(len(labels), (N,), False, probs)
    # Use N digits whose labels are different from composite digit's
    elif background == 'wrong_label':
        probs = [0. if y == labels[idx] else 1. for y in labels]
        probs = [p / sum(probs) for p in probs]
        inds = np.random.choice(len(labels), (N,), False, probs)
    
    return inds


# Create a composite of image dataset[idx] by tiling images specified by the background parameter
def make_tile_image(dataset, labels, idx, background='self', ntile=8):
    # Select n*n digits to make up composite digit
    inds = select_indices(labels, idx, background, ntile*ntile)
    
    xlist = [dataset[ind] for ind in inds] # List of images
    xrows = [np.concatenate(tuple(xlist[i*ntile:(i+1)*ntile]), axis=1) for i in range(ntile)] # List of concat'd rows
    xtile = np.concatenate(tuple(xrows), axis=0) # Full tiled image
    
    if xtile.shape[0] != 224:
        xtile = cv2.reshape(xtile, dsize=(224,224))
    
    # Subtract the original image
    xtile = np.minimum(xtile, cv2.resize(dataset[idx], dsize=(224,224)))
    
    return xtile


# Create a composite of image dataset[idx] by randomly placing images specified by
#    the background parameter
def make_jumbled_image(dataset, labels, idx, background='self'):
    # Resize target image and start blank canvas
    field = cv2.resize(dataset[idx], dsize=(224,224))
    img = np.zeros_like(field, dtype=np.uint8)
    
    # Slim the digit to make the final image more legible
    field = minpool_field(field, 20)
    # Track original footprint of digit
    orig = np.sum(field)
    
    # Choose indices of component images (30 is a reasonable upper limit)
    inds = select_indices(labels, idx, background, 30)
    
    # Add component digits one by one
    for ind in inds:
        img, field = add_component_digit(img, field, dataset[ind])
        if np.sum(field) < orig * 0.01:             #TODO: Tune this threshold
            break
    return img


# Replace each pixel in the field with the darkest pixel no more than rad pixels away in
#   either direction
def minpool_field(field, rad):
    result = np.zeros_like(field)
    H,W = field.shape
    for r in range(0,H):
        for c in range(0,W):
            r0 = max(0,r-rad//2)
            c0 = max(0,c-rad//2)
            r1 = min(H,r+rad//2+1)
            c1 = min(W,c+rad//2+1)
            result[r,c] = np.min(field[r0:r1,c0:c1])
    return result


# Add a single component image to the composite at a random location
def add_component_digit(img, field, component):
    # Make sure there are places left to put the digit
    if not np.any(field):
        return img, field
    
    # Select a point in the image
    r,c = choose_random_point(field)
    
    # Find footprint of component
    h,w = component.shape
    H,W = img.shape
    r0 = min(H-h, max(0, r-(h//2)))
    c0 = min(W-w, max(0, c-(w//2)))
    
    # Add component to composite
    img[r0:r0+h, c0:c0+w] = np.maximum(img[r0:r0+h, c0:c0+w], component)
    
    # Remove footprint from field
    field[r0:r0+h, c0:c0+w] = 0
    
    return img,field


# Choose a random point within the remaining composite field
def choose_random_point(field):
    # Reshape into 1D array
    shape = field.shape
    field = field.reshape([-1,])
    
    # Transform a uniform random point choice into the proper distribution by inverting
    #   the cumulative distribution of the field
    threshold = random.random() * field.sum()
    idx = sum(np.cumsum(field) < threshold)
    
    # Convert into 2D indices
    r = idx // shape[1]
    c = idx % shape[0]
    
    return r,c

    
# Data-parallel wrapper for make_tile_image() and make_jumbled_image()
def wrap_parallel(func, worker_id, num_workers, **kwargs):
    print('[Worker {}/{}] Started'.format(worker_id+1, num_workers))
    # Total images to divide up
    N = len(kwargs['dataset'])
    # Images per worker
    chunk = (N + num_workers - 1) // num_workers # Divide by num_workers, round up
    # Indices of this worker's chunk
    start = chunk*worker_id
    end = min(N, chunk*(worker_id+1))
    # Process chunk
    results = np.zeros((end-start,224,224), dtype=np.uint8)
    for i in range(start,end):
        kwargs['idx'] = i
        results[i-start,:,:] = func(**kwargs)
        #if (i+1) % (chunk//10) == 0:
        print('[Worker {}/{}] Processed {}/{} (index {} in [{},{}))'.format(worker_id+1, num_workers, i-start+1, end-start, i, start, end))
    # Return results
    return start, end, results


# Load a dataset from a .npy file saved by make()
def load(background='self', construction='tilex8'):
    fname = 'mnist_mosaic_{}_{}.npy'.format(background, construction)
    fname = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fname):
        print('No file \'{}\' to load. Call make(\'{}\', \'{}\') first.'.format(fname, background, construction))
        return None, None, None, None
    # Load file
    with open(fname, 'rb') as f:
        mnist_mos = np.load(f, allow_pickle=True).item()
    return mnist_mos['training_images'], mnist_mos['training_labels'], mnist_mos['testing_images'], mnist_mos['testing_labels']

if __name__ == '__main__':
    make('correlated', 'jumble')
