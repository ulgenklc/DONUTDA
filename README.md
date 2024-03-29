# DONUTDA
A python based unsupervised software for Donut-like Object segmeNtation Utilizing Topological Data Analysis.

DONUTDA implements persistent homology to perform image analysis on biomedical image data. Taking a 2d-grayscale image as an input, there are four easy steps for algorithm to spit out the desired masks of the regions of interest(ROI).

Biomedical image analysis in scientific study is necessary to enhance our understanding of biological
systems. Here, we introduce a Python GUI that performs cell localization and segmentation
based on techniques from topological data analysis (TDA) such as persistent homology (PH). The
GUI is designed to be used by those without a mathematical background in TDA and has four
easy steps to perform cell detection. After uploading the original image, users can employ simple
enhancing options to make regions of interest (ROI) clearer in the image. Next, the user chooses
the ROI type to be captured as being either filled (in which case localization is performed using
0-dim PH) or donut-like (in which case segmentation can be performed using 1-dim PH). After
the homology generators are found, a threshold is chosen either manually or automatically via a
pre-determined heuristic. Finally, the user is given the option to refine the amount of detected cells
based on certain geometric parameters such as cell size, cell circularity, or convexity, or to remove
unwanted false ROI. The GUI also offers the user the ability to export the detected ROI to ImageJ
readable files in order to integrate with other image processing pipelines.

![GUI](https://github.com/ulgenklc/DONUTDA/blob/master/data_images/eztda_gui_abst.png)

## INSTALLATION

type `pip install donutda` in your command line

## GUI

Download the repository, go to the path of the file you downloaded in your command line and type `python donutda.py ` 

![lookup](https://github.com/ulgenklc/DONUTDA/blob/master/data_images/lookup_roi.png)

**A** Original image. Use provided preprocessing options or ImageJ. Jump
to next panel by ’Pre-Process’. **B** Pre-Processed image. Choose the ROI type then ’Find ROIs’. **C**
Raw cycles found. Choose either manual or automated thresholding option and hit ’Threshold’. **D**
Final contours drawn on the original image. ’Clean’ the cycle array by choosing some geometric
features. ’Draw’ the final contours. Additionally, you can also mark unwanted individual ROI's as false positives by 'Look Up ROIs'. 

After you are done with the analysis, you can save the individual masks in .tif format to the desired directory.

### References
--


