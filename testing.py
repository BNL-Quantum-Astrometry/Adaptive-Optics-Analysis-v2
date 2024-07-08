import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, BiweightLocationBackground
from photutils.detection import DAOStarFinder

image_path = 'C:/Users/Alex/Desktop/2024-06-27/5ms_sky_test1_fibercam/22_53_42/5ms_sky_test1_fibercam_00001.fits'
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

header = fits.Header()
header['OBJECT'] = 'Example Object'
header['DATE'] = '2024-07-02'

fits.writeto('output.fits', image_clipped, header, overwrite=True)