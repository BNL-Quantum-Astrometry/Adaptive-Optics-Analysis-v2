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
import numpy as np

if __name__ == '__main__':
    print('Testing')

def calculate_light_through_aperture_vectorized(data, centroid_x, centroid_y, radius, subdivisions):
    # Setup for vectorization
    subpixel_size = 1.0 / subdivisions
    subpixel_offsets = np.linspace(-0.5 + 0.5 * subpixel_size, 0.5 - 0.5 * subpixel_size, subdivisions)
    # Mesh of pixel center coordinates
    pixel_centers_x, pixel_centers_y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    # Distances from pixel centers to centroid
    distances_to_centroid = np.sqrt((pixel_centers_x - centroid_x) ** 2 + (pixel_centers_y - centroid_y) ** 2)
    # Identify fully within, border, and outside pixels
    fully_within_circle = distances_to_centroid <= (radius - 1)
    at_border = (distances_to_centroid <= radius) & ~fully_within_circle
    # Calculate total light intensity for fully within pixels
    total_light_intensity = np.sum(data[fully_within_circle])

    # Process border pixels
    for i, j in zip(*np.where(at_border)):
        # Calculate subpixel centers
        subpixel_centers_x = j + subpixel_offsets
        subpixel_centers_y = i + subpixel_offsets
        # distances from subpixel center to centroid
        subpixel_distances_to_centroid = np.sqrt((subpixel_centers_x - centroid_x) ** 2 + (subpixel_centers_y[:, None] - centroid_y) ** 2)
        # Determine subpixels within circle
        within_circle = subpixel_distances_to_centroid <= radius
        # Overlap fraction for border pixel
        overlap_fraction = np.mean(within_circle)
        # Update the total light intensity
        total_light_intensity += data[i, j] * overlap_fraction
    return total_light_intensity