import os
from astropy.io import fits
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import concurrent
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from astropy.stats import SigmaClip
from photutils.background import Background2D, BiweightLocationBackground, ModeEstimatorBackground, MedianBackground
from photutils.detection import DAOStarFinder
from astropy.visualization import ImageNormalize, SqrtStretch
import math

if __name__ == '__main__':
    print('Testing')