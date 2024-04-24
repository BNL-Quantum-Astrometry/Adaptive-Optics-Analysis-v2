
import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Circle
from matplotlib import cm
from astropy.stats import SigmaClip
import imageio
from photutils.background import Background2D, ModeEstimatorBackground, BiweightLocationBackground
from photutils.detection import DAOStarFinder
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import random
import warnings


# Fits handling
def read_fits_data(folder_path):
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.fits'):
            file_path = os.path.join(folder_path, file_name)
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                data_list.append(data)
    return data_list

def aggregate_data(data_list):
    return np.sum(data_list, axis=0)

# Centroid finder and fixed radius aperture analysis
def mark_centroid_and_calculate_percentage(data, radius):

    # Background subtraction   
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = ModeEstimatorBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_subtracted = data - bkg.background
        
    # Normalize data
    norm = ImageNormalize(stretch=SqrtStretch())
    norm_data = norm(data_subtracted)
        
    # DAOStarFinder to find 'stars' in the image
    daofind = DAOStarFinder(fwhm=5, threshold=4*bkg.background_rms_median)
    sources = daofind(data_subtracted)
    if sources is None or len(sources) == 0:
        return None
        
    # Identify brightest source
    brightest_source = sources[np.argmax(sources['peak'])]

    # Get the coordinates of brightest source
    x_centroid, y_centroid = int(brightest_source['xcentroid']), int(brightest_source['ycentroid'])

    
    # Meshgrid for distances from the centroid
    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_centroid = np.sqrt((X - x_centroid)**2 + (Y - y_centroid)**2)
    
    # Mark pixels within and at the border
    fully_within_circle = dist_from_centroid <= (radius - 1)
    fully_within_max_circle = dist_from_centroid <= (75)
    at_border = (dist_from_centroid > (radius - 1)) & (dist_from_centroid <= (radius + 1))

    # Total light intensity for fully within pixels
    total_light_intensity = np.sum(data_subtracted[fully_within_circle])

    # Light intensity for border pixels using subpixels
    total_light_intensity += calculate_light_through_aperture_vectorized(data_subtracted, x_centroid, y_centroid, radius, subdivisions, at_border)

    # Percent capture
    percentage_within_circle = total_light_intensity / np.sum(data_subtracted) * 100

    # Generate mask
    mask = np.zeros_like(data_subtracted, dtype=bool)
    mask[fully_within_circle] = True
    mask[at_border] = True

    return x_centroid, y_centroid, percentage_within_circle, mask

def calculate_light_through_aperture_vectorized(data, centroid_x, centroid_y, radius, subdivisions, at_border):
    subpixel_size = 1.0 / subdivisions
    subpixel_offsets = np.linspace(-0.5 + 0.5 * subpixel_size, 0.5 - 0.5 * subpixel_size, subdivisions)
    total_light_intensity = 0

    # Process border pixels
    for i, j in zip(*np.where(at_border)):
        subpixel_centers_x = j + subpixel_offsets
        subpixel_centers_y = i + subpixel_offsets
        subpixel_distances_to_centroid = np.sqrt((subpixel_centers_x - centroid_x) ** 2 + (subpixel_centers_y[:, None] - centroid_y) ** 2)
        within_circle = subpixel_distances_to_centroid <= radius
        overlap_fraction = np.mean(within_circle)
        total_light_intensity += data[i, j] * overlap_fraction

    return total_light_intensity


def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()

def fit_gaussian(data):
    x = np.linspace(0, data.shape[1] - 1, data.shape[1])
    y = np.linspace(0, data.shape[0] - 1, data.shape[0])
    x, y = np.meshgrid(x, y)

    # x and y coordinates of centroid... center of mass doesnt work as well as DAO, FYI
    centroid_y, centroid_x = center_of_mass(data)

    initial_guess = (159271584, centroid_x, centroid_y, 3, 4, -0.49, 0)
    bounds = ([0, 0, 0, 0, 0, -np.pi/4, 0], [np.inf, data.shape[1] - 1, data.shape[0] - 1, 100, 100, np.pi/4, 1])
    popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), data.ravel(), p0=initial_guess, bounds=bounds, maxfev=1000000)
    return popt

"""
Path for guidecam:
/Users/owenp/BNL Misc./Coding/fits_stuff/Guidcam_turb_320x240_10s/14_24_58

Initial guess for guidecam:
(6153376, centroid_x, centroid_y, 4, 6, -0.59, 0)

x0 = 185
y0 = 70
"""

"""
Path for collcam:
/Users/owenp/BNL Misc./Coding/fits_stuff/Collcam_turb_320x240_10s/14_31_11

Initial guess for collcam:
(159271584, centroid_x, centroid_y, 3, 4, -0.49, 0)

x0 = 139
y0 = 120
"""

def create_gaussian_image(popt, shape=(240, 320)):
    x = np.linspace(0, shape[1] - 1, shape[1])
    y = np.linspace(0, shape[0] - 1, shape[0])
    x, y = np.meshgrid(x, y)
    data_fitted = gaussian_2d((x, y), *popt).reshape(shape)
    return x, y, data_fitted

def save_gaussian_data_image(x, y, image, popt, output_path, red_chi_squared):
    fig, ax = plt.subplots()
    cax = ax.imshow(image, cmap='viridis', extent=(x.min(), x.max(), y.max(), y.min()))
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    
    fig.colorbar(cax, ax=ax, label='Intensity')

    stats_text = (f'Amplitude: {popt[0]:.2f}\n'
                  f'Center X: {popt[1]:.2f}\n'
                  f'Center Y: {popt[2]:.2f}\n'
                  f'Sigma X: {popt[3]:.2f}\n'
                  f'Sigma Y: {popt[4]:.2f}\n'
                  f'Theta: {popt[5]:.2f}\n'
                  f'Offset:{popt[6]:.2f}\n'
                  f'Reduced Chi^2: {red_chi_squared:.2f}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    gauss_output_path = os.path.join(output_path, 'collcam_gauss.jpg')

    plt.savefig(gauss_output_path)
    plt.close()

def save_aggregate_data_image(aggregate_data, output_path):
    # Normalize aggregate data 
    max_value = np.max(aggregate_data)
    if max_value > 0:
        normalized_data = aggregate_data / max_value
    else:
        normalized_data = aggregate_data 

    fig, ax = plt.subplots(figsize= (5,5), constrained_layout=True)
    centroid_x, centroid_y, percentage, mask = mark_centroid_and_calculate_percentage(normalized_data, radius=20)
    half_size = 30  
    x_min = max(0, int(centroid_x - half_size))
    x_max = min(normalized_data.shape[1], int(centroid_x + half_size))
    y_min = max(0, int(centroid_y - half_size))
    y_max = min(normalized_data.shape[0], int(centroid_y + half_size))

    pad_left = int(half_size - (centroid_x - x_min))
    pad_right = int(half_size - (x_max - centroid_x))
    pad_top = int(half_size - (y_max - centroid_y))
    pad_bottom = int(half_size - (centroid_y - y_min))

    data_cropped = normalized_data[y_min:y_max, x_min:x_max]
    data_cropped = np.pad(data_cropped, ((pad_bottom, pad_top), (pad_left, pad_right)), 'constant', constant_values=np.min(normalized_data))

    extent = [0, 60, 0, 60] 
    im = ax.imshow(data_cropped, cmap='viridis', origin='lower', extent=extent)

    ax.set_title('Lab Control', fontsize=20)
    ax.set_xlabel('X Pixel', fontsize=15)
    ax.set_ylabel('Y Pixel', fontsize=15)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity (Norm)', fontsize=20)
    cbar.ax.tick_params(labelsize=20, )
    cbar.set_ticks([0.2,0.4,0.6,0.8,1.0])
    cbar.set_ticklabels([0.2,0.4,0.6,0.8,1.0])
    ax.tick_params(axis='both', which='major', labelsize=15, length=8, width=2) 

    aggregate_data_image_path = os.path.join(output_path, 'LABgc_aggregate_data.svg')

    plt.savefig(aggregate_data_image_path)

def save_aggregate_data_image_with_circle(aggregate_data, output_path):
    centroid_x, centroid_y, percentage, mask = mark_centroid_and_calculate_percentage(aggregate_data)
    
    fig, ax = plt.subplots()

    im = ax.imshow(aggregate_data, cmap='viridis', origin='lower')
    
    # Overlay circular area
    ax.contour(mask, colors='red', linewidths=1.5)
    
    ax.text(0.05, 0.95, f'{percentage:.2f}% within circle', transform=ax.transAxes, color='white')
    
    ax.set_title('Aggregated FITS Data Aperature Analysis')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    fig.colorbar(im, ax=ax, label='Intensity')

    enhanced_image_path = os.path.join(output_path, 'collcam_aggregate_data_with_circle.jpg')
    plt.savefig(enhanced_image_path)
    plt.close()

def prepare_3d_gaussian_data(popt, shape=(240, 320)):
    x = np.linspace(0, shape[1] - 1, shape[1])
    y = np.linspace(0, shape[0] - 1, shape[0])
    Xmesh, Ymesh = np.meshgrid(x, y)
    
    # Flatten Xmesh and Ymesh for input to gaussian_2d then reshape back to 2D for 3D plotting
    Zn = gaussian_2d((Xmesh.ravel(), Ymesh.ravel()), *popt).reshape(Xmesh.shape)
    
    return Xmesh, Ymesh, Zn

def save_3d_gaussian_image(Xmesh, Ymesh, Zn, output_path):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    # Plot surface
    surf = ax.plot_surface(Xmesh, Ymesh, Zn, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_zlim(0, np.max(Zn)+2)

    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Visualization of Gaussian Fit')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig(os.path.join(output_path, 'collcam_3d_gaussian.jpg'))
    plt.close()

def save_3d_aggregate_data_image(aggregate_data, output_path):
    x = np.linspace(0, aggregate_data.shape[1] - 1, aggregate_data.shape[1])
    y = np.linspace(0, aggregate_data.shape[0] - 1, aggregate_data.shape[0])
    Xmesh, Ymesh = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    surf = ax.plot_surface(Xmesh, Ymesh, aggregate_data, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_zlim(0, np.max(aggregate_data) + np.max(aggregate_data) * 0.1)  # 10% headroom above the max value for better visualization
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    

    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Visualization of Aggregate Data')

    plt.savefig(os.path.join(output_path, 'collcam_3d_aggregate_data.jpg'))
    plt.close()


# STATISTICS
def calc_red_chi_squared(observed, expected):
    # Ensure non-zero
    nonzero_mask = (observed != 0) & (expected != 0)

    chi_squared = np.sum(((observed[nonzero_mask] - expected[nonzero_mask]) ** 2) / expected[nonzero_mask])
    
    degrees_of_freedom = np.count_nonzero(nonzero_mask) - len(popt)  
    red_chi_squared = chi_squared / degrees_of_freedom

    return red_chi_squared


def plot_residuals(observed, expected, output_path):
    residuals = observed - expected

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(residuals, cmap='coolwarm', origin='lower')
    ax.set_title('Residuals: Observed - Gaussian Fit')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    fig.colorbar(cax, ax=ax, label='Residual Value')

    residuals_output_path = os.path.join(output_path, 'residuals.jpg')

    plt.savefig(residuals_output_path)
    plt.close()

def mark_centroid_and_calculate_percentage(data, radius=3):
    norm_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    norm_data = np.nan_to_num(norm_data)
    threshold = np.mean(norm_data) + 3 * np.std(norm_data)
    bright_region = norm_data > threshold
    centroid_y, centroid_x = center_of_mass(bright_region)
    
    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_centroid = np.sqrt((X - centroid_x)**2 + (Y - centroid_y)**2)
    
    mask = dist_from_centroid <= radius
    
    percentage_within_circle = np.sum(data[mask]) / np.sum(data) * 100
    
    return centroid_x, centroid_y, percentage_within_circle, mask


def estimate_background(data):
    return np.median(data)

def subtract_background(data, background_level):
    return data - background_level





folder_path = '/Users/owenp/BNL Misc./Coding/fits_stuff/BEST_Lab_Tst/Guidecam_10s_7/14_01_22'
output_path = '/Users/owenp/BNL Misc./Coding/fits_stuff/gifs_and_images/Poster_Data/Part_1'

subdivisions = 50

data_list = read_fits_data(folder_path)
aggregated_data = aggregate_data(data_list)

background_level = estimate_background(aggregated_data)
aggregated_data_subtracted = subtract_background(aggregated_data, background_level)

popt = fit_gaussian(aggregated_data_subtracted)

x, y, gaussian_image = create_gaussian_image(popt)
Xmesh, Ymesh, Zn = prepare_3d_gaussian_data(popt)

red_chi_squared = calc_red_chi_squared(aggregated_data_subtracted, gaussian_image)

#plot_residuals(aggregated_data_subtracted, gaussian_image, output_path)
#save_gaussian_data_image(x, y, gaussian_image, popt, output_path, red_chi_squared)
save_aggregate_data_image(aggregated_data_subtracted, output_path)
#save_aggregate_data_image_with_circle(aggregated_data_subtracted, output_path)
#save_3d_gaussian_image(Xmesh, Ymesh, Zn, output_path)
#save_3d_aggregate_data_image(aggregated_data, output_path)