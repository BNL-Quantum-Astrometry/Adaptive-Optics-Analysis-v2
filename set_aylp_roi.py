'''
created by Alex Gleason

when running the FSM with anyloop, we really want to be getting >100 Hz
because the write speed on the laptop is kinda slow you have to use a smallish roi to get that speed
as you can imagine, it would be a massive pain to have to set that in the field every time by guessing pixel values and changing the json
this script will automatically location the brightest object in field and center of roi of a set size around it

as input you must give the file location of a fits image of the star, i use firecapture on the laptop to take this
arguments are image path first, json path second

ex:
python set_aylp_roi.py capture/image.fits fsmfeedback.json
'''

import sys
import numpy as np
import math
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, BiweightLocationBackground
from photutils.detection import DAOStarFinder


def replace_line(file_path, search_string, replacement):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    found = False
    for i, line in enumerate(lines):
        if search_string in line:
            lines[i] = replacement + '\n'  # Replace the line

            found = True
            break
    
    if found:
        # Write the modified lines back to the file
        with open(file_path, 'w') as file:
            file.writelines(lines)


def main(image_path, conf_path):
    with fits.open(image_path) as hdul:
        image = hdul[0].data

    # Background estimation and subtraction
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = BiweightLocationBackground()
    bkg = Background2D(image, (50,50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    background_rms_median = bkg.background_rms_median + 0.01
    data_subtracted = image - bkg.background
    # Clip so no negatives
    image_clipped = np.clip(data_subtracted, a_min=0, a_max=None)

    daofind = DAOStarFinder(fwhm=3, threshold=1.5*background_rms_median)
    sources = daofind(image_clipped)

    if sources is None or len(sources) == 0:
        raise ValueError("No sources found. Consider adjusting DAOStarFinder parameters.")

    # Identify brightest source
    brightest_source = sources[np.argmax(sources['peak'])]

    # Coordinates for brightest source
    centroid_x, centroid_y = brightest_source['xcentroid'], brightest_source['ycentroid']

    centroid_y = math.floor(centroid_y)
    centroid_x = math.floor(centroid_x)

    start_y = centroid_y - (roi_dim / 2) if centroid_y - (roi_dim / 2) > 0 else 0
    start_x = centroid_x - (roi_dim / 2) if centroid_x - (roi_dim / 2) > 0 else 0
    replace_line(conf_path, "roi_start_y", "\t\t\"roi_start_y\": " + str(start_y) + ",")
    replace_line(conf_path, "roi_start_x", "\t\t\"roi_start_x\": " + str(start_x) + ",")
    replace_line(conf_path, "roi_height", "\t\t\"roi_height\": " + str(roi_dim) + ",")
    replace_line(conf_path, "roi_width", "\t\t\"roi_width\": " + str(roi_dim)) #no comma because its the last param in block
    replace_line(conf_path, "region_height", "\t\t\"region_height\": " + str(roi_dim) + ",")
    replace_line(conf_path, "region_width", "\t\t\"region_width\": " + str(roi_dim) + ",")

roi_dim = 256

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])