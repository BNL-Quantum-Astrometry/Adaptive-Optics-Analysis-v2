import os
import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
import imageio
from concurrent.futures import ProcessPoolExecutor
from photutils.background import Background2D, ModeEstimatorBackground, BiweightLocationBackground
from photutils.detection import DAOStarFinder
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize

output_directory = 'C:/Users/Alex/Documents/SULI research/output'
source_directory = 'C:/Users/Alex/Desktop/SharpCap Captures/2024-06-17/Capture/11_45_20'
gif_filename = 'lab_guidecam_turb_centertracing.gif'
num_workers = 8

os.makedirs(output_directory, exist_ok=True)

def estimate_background(data):
    return np.median(data)

def subtract_background(data, background_level):
    return data - background_level


def process_fits_and_mark_centroid(full_path):
    try:
        with fits.open(full_path) as hdul:
            data = hdul[0].data
        # Background subtraction
        sigma_clip = SigmaClip(sigma=3)
        bkg_estimator = ModeEstimatorBackground()
        bkg = Background2D(data, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        data_subtracted = data - bkg.background
        
        # Normalize data
        norm = ImageNormalize(stretch=SqrtStretch())
        norm_data = norm(data_subtracted)
        
        # DAOStarFinder to find 'stars' in image
        daofind = DAOStarFinder(fwhm=5, threshold=4*bkg.background_rms_median)
        sources = daofind(data_subtracted)
        if sources is None or len(sources) == 0:
            return None
        
        # Keep only brightest source found by DAO
        brightest_source = sources[np.argmax(sources['peak'])]
        
        # Norm to RGB
        rgb_image = np.stack([norm_data] * 3, axis=-1)
        
        # Coordinates of brightest source
        x_centroid, y_centroid = int(brightest_source['xcentroid']), int(brightest_source['ycentroid'])
        
        # Centroid mark
        rgb_image[max(y_centroid-1, 0):y_centroid+2, max(x_centroid-1, 0):x_centroid+2, :] = [1, 0, 0]
        
        # RGB to uint8
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image_uint8

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == '__main__':
    fits_paths = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if f.endswith('.fits')]

    # Parallel processing
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        frames = list(executor.map(process_fits_and_mark_centroid, fits_paths))

    frames = [frame for frame in frames if frame is not None]

    # Build gif
    if frames:
        gif_path = os.path.join(output_directory, gif_filename)
        imageio.mimsave(gif_path, frames, 'GIF', duration=0.5)
        print(f"GIF saved successfully at {gif_path}.")
    else:
        print("No FITS files processed.")
