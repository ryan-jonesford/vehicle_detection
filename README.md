# **Vehicle Detection Project**

This project creates a pipeline for processing frames from a video to detect and draw bounding boxes around vehicles. It uses a linear SVM for classification and datasets from Udacity,  GTI vehicle image database, and the KITTI vision benchmark suite.  This project is part of the Udacity Self-driving car Nanodegree program, some functions contained in this project are borrowed and/or reproduced from there. 

---
#### Files
README - this file
vehicle_detection.py - Main program
vehicle_detection.ipynb - Jupyter notebook used for demonstrating individual functions
pantry/ - Folder for pickled data
     features.p - pickled feature data
     classifier.p - pickled classifier
     scaler.p - pickled scaler
output_videos/  - folder containing video annotated by vehicle_detection.py  
     project_video.mp4 - anotated output video


### Histogram of Oriented Gradients (HOG)

#### HOG and Color feature extraction
##### **HOG**

I started by reading in all the `vehicle` and `non-vehicle` images.  An example of each can be seen in code cell 6 of vehicle_detection.ipynb along with some general information on the dataset. 

I tried various combinations of parameters and found that the important car features can be found within about a 20x20 box. So I chose 20x20 for my pixels per cell. Through trial and error I found that having 2 cells per block and 9 orientations produced the lowest trainer error.

These values where declared as globals in lines 20-42 in vehicle_detection.py to keep consistency throughout the program.

Another step I took to help the HOG extraction was to apply a gaussian blur to each image to smooth out camera defects. 

An example of the HOG extraction on both datasets can be seen in code cell 7 in vehicle_detection.ipynb

##### **Spacial/Color**
I reasoned that since most car colors don't match the natural environment, I could pool the color features into a very small area, that I chose to be 10x10 and then bin the color data into 6 bins, since most cars are just slight variations of primary colors. However I needed to use trial and error to decide on using the YCrCb color space. 

Between the spacial binning and the large pixels per cell, it allowed me to have a small feature space which allows for faster image processing. 

In code cell 9 of vehicle_detection.ipynb there is a graph of my feature space before an after normalization. 

The feature extraction pipeline was put into a single function "extract_features (line 116 vehicle_detection.py). 


#### Training a classifier 

I trained a linear SVM by first retrieving the dataset (line 413 vehicle_detection.py), extracting the features (explained above), then normalizing and splitting the feature space dataset into a training and testing set (line 454 vehicle_detection.py). I found my test accuracy to be about 98% taking only about 32seconds to train the classifier. 

### Finding Cars

####  Search Area

I used a sliding windows technique to traverse images to find cars. This is all done in the function find_cars (line 213 vehicle_detection.py). The function expects you to define a y-axis start and end position, and a scale at which you want to search at. A scale of 1 corresponds to a window size of 64x64, the same as the training image sizes. I found that I could get relatively good detection with three different window sizes (with 2 cells per step) and search areas (line 474 vehicle_detection.py, code cell 13 vehicle_detection.ipynb). 


#### Pipeline On Test Images

Code cell 13 in vehicle_detection.ipynb shows the output from my pipeline with an associated heatmap. The heatmap takes boxes and increments the elements on a 2d zero array by 1 where the boxes are located. 

---

### Video Implementation

#### Link to annotated video output. 

Here's a [link to my video.](https://github.com/ryan-jonesford/vehicle_detection/blob/master/output_videos/proj_video_out.mp4)
It performs okay, not as well as I'd like, with more false positives than I'd accept if I had more time and support. However I feel the vehicles are detected well, even with the boxes being a tad unstable. I've also added the heatmap output to the top right of the video to aid in general understanding of the bounding boxes.


#### False Positive Rejections and Filtering

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

These labels where then saved in my class, vehicle_tracker, for 15 frames. These labels where then overlaid on each other to create another heatmap, with a threshold of half the saved frame count. The class is located in line 280 of vehicle_detection.py.
The pipeline for this is all done in the main function of the program, detector (line 356 vehicle_detection.py). 

Code Cell 15 in vehicle_detection.ipynb shows an example result. It shows the heatmap from a series of frames of video in the upper right corner, which is the result of `scipy.ndimage.measurements.label()` with the bounding boxes then overlaid.

---

### Discussion

#### Extra Info / Problems / Issues 

I ran into many issues during this project. The least of them being unable to get rid of false positives while at the same time detecting the vehicles that were present in the frame and being unable to test it fast enough. At fist my feature length was around 8k and it took up to 2.5 seconds per frame with the windows that I specified, making for a very long wait to see if any changes I made, made a difference.

One of my main goals became to reduce my feature space size as much as possible, while keeping the classifier accuracy above 97.5%, so I could detect vehicles and test changes in a timely manner. I was able to reduce my feature space size by 10 fold, netting me a speed increase to 3 frames per second. I also tried to keep the number of windows I was using low to keep the speed up, which also might have contributed to false positives and unstable bounding boxes in my result. 

With the relatively low performance of this pipeline I would expect it to fail in multiple situations such as inclement weather, hills, and night. I also don't think this would be a very good pipeline for a production vehicle, even if it worked perfectly, due to it being a resource intensive pipeline. It is just too slow for real-time applications on affordable hardware. 

[Rubric](https://review.udacity.com/#!/rubrics/513/view) 

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
