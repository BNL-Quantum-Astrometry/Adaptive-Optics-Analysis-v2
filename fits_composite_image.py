import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

source_directory = '/Users/owenp/BNL Misc./Coding/fits_stuff/Fst_On_Sky_Tst/Sirius_guidecam_nofilter_5ms_1bin/21_50_51'
output_directory = '/Users/owenp/BNL Misc./Coding/fits_stuff'
os.makedirs(output_directory, exist_ok=True)

data_list = []

for filename in os.listdir(source_directory):
    if filename.endswith(".fits"):
        file_path = os.path.join(source_directory, filename)

        try:
            with fits.open(file_path) as hdul:
                data = hdul[0].data
                data_list.append(data)

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

if data_list:
    # Generate mean data composite
    stacked_data = np.dstack(data_list)
    average_data = np.mean(stacked_data, axis=2)

    # Normalize data
    norm_data = (average_data - np.min(average_data)) / (np.max(average_data) - np.min(average_data))

    plt.figure()
    plt.imshow(norm_data, cmap='gray', origin='lower') 
    plt.axis('on') 
    plt.tight_layout()

    svg_filename = os.path.join(output_directory, 'sky_guidecam_avgimg.svg')
    plt.savefig(svg_filename, format='jpg', bbox_inches='tight', pad_inches=0.0)
    plt.close()

    print(f"Averaged image saved as SVG successfully at {svg_filename}.")
else:
    print("No FITS files processed.")

print("Processing complete.")
