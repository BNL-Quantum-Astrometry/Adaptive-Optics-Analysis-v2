'''
written by Owen Leonard

Creates a gif of input fits files
'''

import os
import numpy as np
from astropy.io import fits
import imageio
from concurrent.futures import ProcessPoolExecutor


num_workers = 8
output_directory = 'C:/Users/Alex/Documents/SULI research/output'
source_directory = 'C:/Users/Alex/Desktop/SharpCap Captures/2024-06-17/Capture/11_45_20'
gif_filename = 'sky_guidecam_turb.gif'

os.makedirs(output_directory, exist_ok=True)

def process_fits_file(filename):
    file_path = os.path.join(source_directory, filename)
    try:
        with fits.open(file_path) as hdul:
            data = hdul[0].data

        # Normalize data and scale
        norm_data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        norm_data = np.nan_to_num(norm_data)
        scaled_data = (norm_data * 255).astype(np.uint8)
    

        return scaled_data
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        return None

def main(num_workers=4):
    os.makedirs(output_directory, exist_ok=True)
    fits_filenames = [f for f in os.listdir(source_directory) if f.endswith('.fits')]
    
    # Parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(process_fits_file, fits_filenames))

    frames = [frame for frame in frames if frame is not None]

    # Make gif
    if frames:
        gif_path = os.path.join(output_directory, gif_filename)
        imageio.mimwrite(gif_path, frames, format='GIF', fps=30)
        print(f"GIF saved successfully at {gif_path}.")
    else:
        print("No FITS files processed.")

if __name__ == '__main__':
    main()