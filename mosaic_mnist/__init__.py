import sys
import os
packagepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, packagepath)

try:
    from .mosaic import make,load,fetch
    from .pytorch_utils import MnistMosaicDataset,grayscale2color
    sys.path.remove(packagepath)
except ValueError:
    pass

