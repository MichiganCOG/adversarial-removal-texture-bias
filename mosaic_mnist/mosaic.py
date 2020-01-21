import numpy as np
import scipy.sparse as sparse
import os
import cv2
import random
import mnist
import multiprocessing as mp
from matplotlib import pyplot as plt

from urllib import request
import gzip
import shutil

if __name__ != '__main__':
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

'''
Make the mosaic dataset as requested and save it.
Inputs:
 - mode: Determines which component digits are used to make up the composite image
    MODES DETAILED IN PAPER:
    - 'consistent'*: Components are chosen randomly, with higher probability given to images
                    with the same label as the composite image
    - 'inconsistent'*: Components are chosen uniformly at random.
    - 'malicious'*: Same as 'consistent', but higher probability is given to images with a label
                    chosen at random from the set of wrong labels.
    OTHER MODES:
    - 'self': Components are each a copy of the target composite image. Much easier
    - 'same_label': Components are randomly chosen from images with same label as composite image
    - 'different_label': Components are randomly chosen from images with labels different from
                         the composite image's label
    - 'anticonsistent': Same as 'consistent', but with lower probability given to same-label images
 - dset: Which set(s) to make ('train', 'test', or 'both')
 - fname: Filename base to save the dataset(s) to. If absolute, used unchanged. If a filename with no
        path is given, files will be put in DATA_DIR. Default: value of mode.
 - overwrite: If true, overwrites existing files with same name
 - num_workers: Number of parallel workers to do data generation. If -1, uses max available.
'''     
def make(mode='correlated', dset='both', fname=None, overwrite=False, num_workers=-1):
    
    # Validate inputs
    modes = ['self', 'same_label', 'different_label', 'consistent', 'anticonsistent', 'malicious', 'inconsistent']
    assert mode in modes, 'Invalid mode: {} (valid choices are [{}]'.format(mode, ','.join(modes))
    dsets = ['train','test','both']
    assert dset in dsets, 'Invalid dataset: {} (valid choices are [{}]'.format(dset, ','.join(dsets))
    assert num_workers >= -1, 'Invalid number of workers: {}'.format(num_workers)
    # Validate and correct filename base
    fname = mode if fname is None else fname
    if not os.path.isabs(fname) and len(os.path.dirname(fname)) == 0:
        fname = os.path.join(DATA_DIR, fname)
    if fname.endswith(('.np','.npz')):
        fname,ext = os.path.splitext(fname)
    
    # Load MNIST, convert to lists of images
    print('Loading MNIST...')
    xtrain, ytrain, xtest, ytest = mnist.load()
    Ntrain = xtrain.shape[0]
    Ntest = xtest.shape[0]
    xtrain = [xtrain[i,:].reshape([28,28]) for i in range(Ntrain)]
    xtest = [xtest[i,:].reshape([28,28]) for i in range(Ntest)]
    
    # Make training set
    if dset in ('train','both'):
        print('Making training set...')
        trainname = fname if dsets == 'train' else fname + '_train'
        make_dset(xtrain, ytrain, mode, trainname, overwrite, num_workers)
    
    # Make test set
    if dset in ('test', 'both'):
        print('Making test set...')
        testname = fname if dsets == 'test' else fname + '_test'
        make_dset(xtest, ytest, mode, testname, overwrite, num_workers)
    
   
# Make one dataset (train or test) from the original MNIST version
def make_dset(data, labels, mode, fname=None, overwrite=False, num_workers=-1):
    
    data_file = fname if fname.endswith('.npz') else fname + '.npz'
    if os.path.exists(data_file) and not overwrite:
        print('File {} exists. Skipping...'.format(data_file))
        return
    
    # Initialize results
    print('Initializing results...')
    N = len(data)
    data_out = np.zeros([N, 224, 224], dtype=np.uint8)
    labels_out = [None]*N
    
    # Define callback function for parallel workers
    def callback(results):
        data_out[results[0]:results[1],:,:] = results[2]
        labels_out[results[0]:results[1]] = results[3]
    
    # Without parallelization, construct dataset one image at a time
    if num_workers == 1 or num_workers == 0:
        print('Making dataset...')
        print('[0/{}]'.format(N))
        for i in range(N):
            data_out[i,:,:],labels_out[i] = make_one_image(data, labels, i, mode)
            if i % 10 == 9:
                print('[{}/{}]'.format(i+1,N))
    
    # With parallelization, split up dataset evenly and assign each worker a chunk
    else:
        if num_workers == -1:
            num_workers = mp.cpu_count()
        print('Opening pool of {} workers...'.format(num_workers))
        pool = mp.Pool(num_workers)
        
        # Start each worker with a call to helper function make_one_image_chunk
        for worker_id in range(num_workers):
            pool.apply_async(make_one_image_chunk,
                             args=(worker_id, num_workers),
                             kwds={'data': data,
                                   'labels': labels,
                                   'mode': mode},
                             callback=callback)
        pool.close()
        pool.join() # Block execution until all workers are done
    
    # Save the dataset
    save(data_out, labels_out, fname)
        

# Makes one composite image and returns it, along with the label and metadata for it
def make_one_image(data, labels, idx, mode):
    # Resize target image and start blank canvas
    field = cv2.resize(data[idx], dsize=(224,224))
    img = np.zeros_like(field, dtype=np.uint8)
    
    # Slim the digit to make the final image more legible. Tweak this if you want.
    field = minpool_field(field, 20)
    # Track the original total footprint of the digit
    orig_fprint = np.sum(field)
    
    # Choose indices of component images (30 is a reasonable upper limit in practice)
    #    wrong_label only used if mode is 'wrong_label'; otherwise None
    inds, wrong_label = select_indices(labels, idx, mode, 30)
    
    # Initialize metadata entry
    metadata = {'label': labels[idx], 'components': []}
    if mode == 'wrong_label':
        metadata['wrong_label': wrong_label]
    
    # Add components one by one
    for ind in inds:
        img, field, metadata = add_component_digit(img, field, data[ind], labels[ind], ind, metadata)
        if np.sum(field) < orig_fprint * 0.01: # Tune this if you want
            break
    return img, metadata


# Apply a minpool operation with square radius to the field
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
def add_component_digit(img, field, component, label, idx, metadata):
    # Make sure there are places left to put the digit
    if not np.any(field):
        return img, field, metadata
    
    # Select a point in the image by reinterpreting the field as a probability distribution
    coord = np.random.choice(224*224, None, False, [f / np.sum(field) for f in field.reshape([-1,])])
    r = coord // field.shape[1]
    c = coord % field.shape[1]
    
    # Find footprint of component
    h,w = component.shape
    H,W = img.shape
    r0 = min(H-h, max(0, r-(h//2)))
    c0 = min(W-w, max(0, c-(w//2)))
    
    # Add component to image
    img[r0:r0+h, c0:c0+w] = np.maximum(img[r0:r0+h, c0:c0+w], component)
    
    # Remove footprint from field
    field[r0:r0+h, c0:c0+w] = 0
    
    # Add component to metadata
    metadata['components'].append({'x': c, 'y': r, 'label': label, 'index': idx})
    
    return img, field, metadata


# Choose N indices of digits to make up the composite digit based on the mode. Output fake_label
#    is None unless mode is 'wrong_label'.
def select_indices(labels, idx, mode, N):

    fake_label = None
    
    # Use the composite digit N times:
    if mode == 'self':
        return [idx]*N
    # (Rest of modes use probabilistic selection)
    # Use N digits of the same label as composite digit
    elif mode == 'same_label':
        probs = [1. if y == labels[idx] else 0. for y in labels]
    # Use N digits of labels different than composite digit's
    elif mode == 'different_labels':
        probs = [0. if y == labels[idx] else 1. for y in labels]
    # Choose digits randomly, favoring same-labeled digits highly
    elif mode == 'consistent':
        probs = [1. if y == labels[idx] else 0.05 for y in labels]
    # Choose digits randomly, favoring differently-labeled digits highly
    elif mode == 'anticonsistent':
        probs = [0.05 if y == labels[idx] else 1. for y in labels]
    # Pick a wrong label, then choose digits as if it were correct and the mode was 'correlated'
    elif mode == 'malicious':
        fake_label = np.random.choice(list(range(labels[idx])) + list(range(labels[idx]+1,10)))
        probs = [1. if y == fake_label else 0.05 for y in labels]
    # Ignore labels and choose randomly. What's life without a little whimsy?
    elif mode == 'inconsistent':
        probs = [1.]*N
    
    # Normalize probabilities and choose digits
    probs = [p / sum(probs) for p in probs]
    inds = np.random.choice(len(labels), (N,), False, probs)
    
    return inds, fake_label


# Call make_one_image several times in serial, processing one chunk of the dataset
def make_one_image_chunk(worker_id, num_workers, **kwargs):
    
    # Total images to divide up
    N = len(kwargs['data'])
    # Images per worker
    chunk_size = (N + num_workers - 1) // num_workers # Div by num_workers, round up
    # Indices of this worker's chunk
    start = chunk_size * worker_id
    end = min(N, chunk_size * (worker_id+1))
    print('[Worker {}/{}] Started: indices {}:{}'.format(worker_id+1, num_workers, start, end-1))
    
    # Process chunk
    data_out = np.zeros((end-start, 224, 224), dtype=np.uint8)
    labels_out = [None]*(end-start)
    for i in range(start,end):
        kwargs['idx'] = i
        data_out[i-start,:,:], labels_out[i-start] = make_one_image(**kwargs)
        print('[Worker {}/{}] Processed {}/{} (index {} in {}:{})'.format(worker_id+1, num_workers, i-start+1, end-start, i, start, end-1))
    
    # Return results
    return start, end, data_out, labels_out
    
    
# Save a dataset into two files: fname.npz and fname.np.
def save(data, labels, fname):
    # Sparsify the dataset to save space
    data = sparse.csr_matrix(data.reshape([data.shape[0],-1]))
    with open(fname + '.npz', 'wb') as f:
        sparse.save_npz(f, data)
    with open(fname + '.np', 'wb') as f:
        np.save(f, labels)


# Load a dataset from two files: fname.npz and fname.np
def load(fname, label_only=False, dense=True):
    # Remove .np or .npz extension
    if fname.endswith(('.np','.npz')):
        fname,ext = os.path.splitext(fname)

    # If fname is absolute, skip this:
    if not os.path.isabs(fname):
        # If files exist as is, skip this:
        if not os.path.exists(fname + '.np') and not os.path.exists(fname + '.npz'):
            # Look for it in DATA_DIR
            fname_ = os.path.join(DATA_DIR, fname)
            if not os.path.exists(fname_ + '.np') and not os.path.exists(fname_ + '.npz'):
                raise FileNotFoundError('Files {}.np and/or {}.npz not found.'.format(fname,fname))
            fname = fname_
    
    # Load data file
    with open(fname + '.npz', 'rb') as f:
        data = sparse.load_npz(f)
    if dense:
        data = data.todense().getA().reshape([-1,224,224]) # Unpack sparse representation
        
    # Load label file
    with open(fname + '.np', 'rb') as f:
        labels = np.load(f)
    if label_only: # Throw out metadata if requested
        labels = [y['label'] for y in labels]
    return (data,labels)


# URLs of the zipped datasets   TODO: Fill these out, then add more
URL_by_dataset = {
    'consistent_train': None,
    'consistent_test': None,
    'inconsistent_train': None,
    'inconsistent_test': None,
    'malicious_train': None,
    'malicious_test': None,
    'self_train': None,
    'self_test': None,
    'same_label_train': None,
    'same_label_test': None,
    'different_label_train': None,
    'different_label_test': None,
    'anticonsistent_train': None,
    'anticonsistent_test': None
}

# Download one or more premade datasets into the specified folder. Pass 'all' to download all
#   currently available, or 'paper' to download the four used in the Adversarial Removal of
#   Texture Bias paper
def fetch(dataset='paper', folder=DATA_DIR, overwrite=False):
    if dataset == 'all':
        dataset = list(URL_by_dataset.keys())
    elif dataset == 'paper':
        dataset == ['consistent_train','consistent_test','inconsistent_test','malicious_test']
    elif not isinstance(dataset, list):
        dataset = [dataset]
    
    os.makedirs(folder, exist_ok=True)
    
    for d in dataset:
        urlbase = URL_by_dataset[d]
        filebase = os.path.join(folder, d)
        if urlbase is None: # In case not all datasets make it online
            continue
        
        if not os.path.exists(filebase + '.npz') or overwrite:
            print('Downloading {}...'.format(urlbase + '.npz'))
            request.urlretrieve(urlbase + '.npz', filebase + '.npz')
        
        if not os.path.exists(filebase + '.np') or overwrite:
            print('Downloading {}...'.format(urlbase + '.np'))
            request.urlretrieve(urlbase + '.np', filebase + '.np')


if __name__ == '__main__':
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    make('consistent')

