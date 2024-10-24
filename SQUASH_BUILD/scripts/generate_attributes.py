import sys
sys.path.append('SQUASH_BUILD/src')
from pathlib import Path
import numpy as np
from numpy import linalg as LA
import os

num_attributes  = 4

# num_vectors     = 10000
# path            = Path('SQUASH_BUILD/datasets/siftsmall/')
# fname           = 'siftsmall'

# num_vectors     = 1000000
# path            = Path('SQUASH_BUILD/datasets/sift1m/')
# fname           = 'sift1m'

# num_vectors     = 1000000
# path            = Path('SQUASH_BUILD/datasets/gist1m/')
# fname           = 'gist1m'

num_vectors     = 10000000
path            = Path('SQUASH_BUILD/datasets/deep10m/')
fname           = 'deep10m'

full_fname      = os.path.join(path, '') + fname + '.af'
mode            = 'random'

with open(full_fname, mode="wb") as f:
    
    for i in range(num_attributes):
    
        if mode == 'random':
            # Generate random floats in the half-open interval [0.0, 1.0)
            attr = np.float32(np.random.random_sample(size=num_vectors))
            f.write(attr)
        
        elif mode == 'gaussian':
            # Generate random floating from the standard (normal) distribution, 0-1
            attr = np.float32(np.random.randn(num_vectors))
            f.write(attr)

        else:
            print("Mode must be random or gaussian")
            exit(1)

        

    
