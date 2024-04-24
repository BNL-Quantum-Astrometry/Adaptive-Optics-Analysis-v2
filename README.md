As of right now the analyses being used to best determine the effectiveness of the first-order AO are frame_comp_aperture_calcs.py, 
perc_comp_aperture_calcs.py, and aperture_groups.py

All .fits image data is processed in frame_comp_aperture_calcs.py. This is where the double optimization for the single-frame aperture analysis exists. 
General and Rudimentary plots are automatically created here.

Confusingly, the actual plotting functions used for the single-frame aperture analysis, are not where that double optimization code is.
Those are located in perc_comp_aperture_calcs.py, which makes a wide variety of single-frame analysis plots. These include simple frame-by-frame representations and percent-capture
comparisons.

The code for the aggregate aperture analysis, data processing and all, is in aperture_groups.py.

