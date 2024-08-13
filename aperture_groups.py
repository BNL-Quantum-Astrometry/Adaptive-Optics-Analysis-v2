'''
Created by Owen Leonard

This script combines images into a single aperature and creates a comparison between the control (no PID) and guide (PID) views to see if
it is getting better or worse. It also overlays rings so you can see the aperature at various radii. 


'''
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



# DATA PROCESSING

# Fits handling
def read_fits_data(source_path, div_fac, batch_size=100,):
    data_list = []
    file_names = [f for f in os.listdir(source_path) if f.endswith('.fits')]
    for i in range(0, len(file_names), batch_size):
        batch = file_names[i:i+batch_size]
        for file_name in batch:
            file_path = os.path.join(source_path, file_name)
            with fits.open(file_path) as hdul:
                data = (hdul[0].data)/div_fac
                data_list.append(data)
    return data_list


def get_aggregate_data(data_list):
    return np.sum(data_list, axis=0)


def get_random_frame_data(data_list):
    return random.choice(data_list)

# Includes pixel subdivision for higher accuracy
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


def mark_centroid_and_calculate_percentage(data, radius):
        
    # Background subtraction
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = ModeEstimatorBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    data_subtracted = data - bkg.background
        
    # Normalize data
    norm = ImageNormalize(stretch=SqrtStretch())
    norm_data = norm(data_subtracted)
        
    # DAOStarFinder to find 'stars' in the image... works a lot better than center of mass especially for actual on sky data with more background
    daofind = DAOStarFinder(fwhm=5, threshold=4*bkg.background_rms_median)
    sources = daofind(data_subtracted)
    if sources is None or len(sources) == 0:
        return None  # No sources found
        
    # Keep only the brightest source found by DAO
    brightest_source = sources[np.argmax(sources['peak'])]
        
        
    # Coordinates for brightest source
    x_centroid, y_centroid = int(brightest_source['xcentroid']), int(brightest_source['ycentroid'])

    
    # Meshgrid for distances from centroid
    Y, X = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_centroid = np.sqrt((X - x_centroid)**2 + (Y - y_centroid)**2)
    
    # Mark pixels within and at the border of the specified radius from the centroid
    fully_within_circle = dist_from_centroid <= (radius - 1)
    at_border = (dist_from_centroid > (radius - 1)) & (dist_from_centroid <= (radius + 1))

    # Calculate the total light intensity for fully within pixels
    total_light_intensity = np.sum(data_subtracted[fully_within_circle])

    # Calculate the light intensity for border pixels using subdivision method
    total_light_intensity += calculate_light_through_aperture_vectorized(data_subtracted, x_centroid, y_centroid, radius, subdivisions, at_border)

    # Percentage within aperture
    percentage_within_circle = total_light_intensity / np.sum(data_subtracted) * 100

    # Mask for visualization 
    mask = np.zeros_like(data_subtracted, dtype=bool)
    mask[fully_within_circle] = True
    mask[at_border] = True

    return x_centroid, y_centroid, percentage_within_circle, mask


# GROUPS OF SPECIFIC RADII BASED ON FRAME_COMP_APERTURE_CALCS.PY

def save_data_images_with_multiple_circles(data_lists_list, radii_list, shared_x_title, shared_y_title, left_titles, top_titles, output_path, file_name):
    # Initiate subplots
    n_rows, n_cols = 2, 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10), sharex='all', sharey='all')
    fig.subplots_adjust(right=0.8, wspace=0.022, hspace=0.1)

    # Normalize data for each plot
    for idx, (data, radii) in enumerate(zip(data_lists_list, radii_list)):
        ax = axs.flat[idx]
        max_pixel_value = np.max(data)
        data_normalized = data / max_pixel_value if max_pixel_value > 0 else data

        # Multiple aperture overlay on each plot
        for radius in radii:
            centroid_x, centroid_y, percentage, mask = mark_centroid_and_calculate_percentage(data_normalized, radius)
            
            half_size = 30
            x_min = max(0, int(centroid_x - half_size))
            x_max = min(data_normalized.shape[1], int(centroid_x + half_size))
            y_min = max(0, int(centroid_y - half_size))
            y_max = min(data_normalized.shape[0], int(centroid_y + half_size))

            pad_left = half_size - (centroid_x - x_min)
            pad_right = half_size - (x_max - centroid_x)
            pad_top = half_size - (y_max - centroid_y)
            pad_bottom = half_size - (centroid_y - y_min)

            data_cropped = data_normalized[y_min:y_max, x_min:x_max]
            data_cropped = np.pad(data_cropped, ((pad_bottom, pad_top), (pad_left, pad_right)), 'constant', constant_values=np.min(data_normalized))
            
            im = ax.imshow(data_cropped, cmap='viridis', origin='lower', extent=[0, 60, 0, 60])
            circle = Circle((30, 30), radius, edgecolor='red', facecolor='none', linewidth=.5)
            ax.add_patch(circle)
            print(f'[MULTI APP] Subplot {idx+1}: {percentage:.2f}% within {radius} pixel radius')

        ax.set_xticks(np.linspace(0, 60, num=7))
        ax.set_yticks(np.linspace(0, 60, num=7))
        ax.tick_params(axis='both', which='major', labelsize=15, length=8, width=2) 

    cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity (Norm)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    fig.text(0.5, 0.04, shared_x_title, ha='center', fontsize=15)
    fig.text(0.04, 0.5, shared_y_title, va='center', rotation='vertical', fontsize=15)

    for ax, side_title in zip(axs[:, 0], left_titles):
        ax.set_ylabel(side_title, rotation=90, size=20, labelpad=10)

    for ax, top_title in zip(axs[0], top_titles):
        ax.set_title(top_title, pad=10, fontsize=20)

    enhanced_image_path = os.path.join(output_path, file_name)
    plt.savefig(enhanced_image_path, dpi=600)
    plt.close()


# RELEVANT INDIVIDUAL RADII

def save_data_images_with_one_app(data_lists_list, radii_list, shared_x_title, shared_y_title, left_titles, top_titles, output_path, file_name):
    # Initiate subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
    fig.subplots_adjust(right=.8, wspace=0.022, hspace=0)

    # Normalize for each plot and plot with aperture overlayed
    for idx, (data, radius) in enumerate(zip(data_lists_list, radii_list)):
        ax = axs.flat[idx]

        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)

        centroid_x, centroid_y, percentage, mask = mark_centroid_and_calculate_percentage(data, radius)

        half_size = 30
        x_min = max(0, int(centroid_x - half_size))
        x_max = min(data.shape[1], int(centroid_x + half_size))
        y_min = max(0, int(centroid_y - half_size))
        y_max = min(data.shape[0], int(centroid_y + half_size))

        pad_left = half_size - (centroid_x - x_min)
        pad_right = half_size - (x_max - centroid_x)
        pad_top = half_size - (y_max - centroid_y)
        pad_bottom = half_size - (centroid_y - y_min)

        data_cropped = data[y_min:y_max, x_min:x_max]
        data_cropped = np.pad(data_cropped, ((pad_bottom, pad_top), (pad_left, pad_right)), 'constant', constant_values=np.min(data))

        extent = [0, 60, 0, 60] 
        im = ax.imshow(data_cropped, cmap='viridis', origin='lower', extent=extent)

        circle = Circle((30, 30), radius, edgecolor='red', facecolor='none', linewidth=.5)
        ax.add_patch(circle)

        ax.set_xticks(np.linspace(0, 60, num=7)) 
        ax.set_yticks(np.linspace(0, 60, num=7)) 
        ax.tick_params(axis='both', which='major', labelsize=15, length=6, width=2) 

        print(f'[SNGL APP] Subplot {idx+1}: {percentage:.2f}% within {radius} pixel radius')

    pos_top_left = axs[0, 0].get_position()
    axs[0, 0].set_position([pos_top_left.x0, pos_top_left.y0-0.04, pos_top_left.width, pos_top_left.height])
    pos_top_right = axs[0, 1].get_position()
    axs[0, 1].set_position([pos_top_right.x0, pos_top_right.y0 - 0.04, pos_top_right.width, pos_top_right.height])
    cbar_ax = fig.add_axes([0.85, 0.23, 0.05, 0.4])
    fig.colorbar(im, cax=cbar_ax, label='Intensity (Norm)')
    cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity (Norm)', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    fig.text(0.46, 0.13, shared_x_title, ha='center', fontsize=15)
    fig.text(0.08, 0.43, shared_y_title, va='center', fontsize=15, rotation='vertical')
    for ax, side_title in zip(axs[:,0], left_titles):
        ax.set_ylabel(side_title, rotation=90, size=20, labelpad=15)
    for ax, top_title in zip(axs[0], top_titles):
        ax.set_title(top_title, size=20, pad=5)
    enhanced_image_path = os.path.join(output_path, file_name)
    plt.savefig(enhanced_image_path, dpi=600)


warnings.filterwarnings("ignore", category=RuntimeWarning) #This was annoying but couldnt figure out how to fix, it didn't cause any issues with the results so I just swept under the rug
subdivisions = 50
cc_source_path = 'C:/Users/Alex/Desktop/SharpCap Captures/2024-08-01/Capture/23_03_53'
gc_source_path = 'C:/Users/Alex/Desktop/SharpCap Captures/2024-08-01/Capture/22_08_10'
output_path =  'C:/Users/Alex/Documents/SULI research/output'


def main(output_path, gc_aggregate_data, cc_aggregate_data, gc_random_frame_data, cc_random_frame_data):
    save_data_images_with_multiple_circles([gc_aggregate_data, gc_random_frame_data, cc_aggregate_data, cc_random_frame_data], [[5, 10, 15] for _ in range(4)],
                                        'X Pixel', 'Y Pixel', ['Control', 'AO-Corrected'], ['Aggregate', 'Single Frame'], output_path, "aggregate_data_multiple_apertures.svg") #USE CC AVG OPTIMAL RADII AT 25,50,75 PERC CAPTURE FOR RADII
    save_data_images_with_one_app([gc_aggregate_data, gc_aggregate_data, cc_aggregate_data, cc_aggregate_data], [1.75, 4, 1.75, 4],
                                        'X Pixel', 'Y Pixel', ['Control', 'AO-Corrected'], ['Fiber', 'Lantern'], output_path, "aggregate_data_single_apertures.svg")

if __name__ == "__main__":

    gc_data_list = read_fits_data(gc_source_path, div_fac=8)
    cc_data_list = read_fits_data(cc_source_path, div_fac=1)
    gc_aggregate_data = get_aggregate_data(gc_data_list)
    cc_aggregate_data = get_aggregate_data(cc_data_list)
    gc_random_frame_data = get_random_frame_data(gc_data_list)
    cc_random_frame_data = get_random_frame_data(cc_data_list)
    main(output_path, gc_aggregate_data, cc_aggregate_data, gc_random_frame_data, cc_random_frame_data)