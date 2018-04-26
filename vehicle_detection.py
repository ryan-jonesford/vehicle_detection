import os
import glob
import pickle
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.ndimage.measurements import label
from collections import deque
import time
import imageio
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


# Define a color space dict. All image reads done by cv2.imread()
CS = {'HSV':    cv2.COLOR_BGR2HSV,
      'YUV':    cv2.COLOR_BGR2YUV,
      'YCrCb':  cv2.COLOR_BGR2YCrCb,
      'LUV':    cv2.COLOR_BGR2LUV,
      'HLS':    cv2.COLOR_BGR2HLS}
# Declare these globally, so there isn't confusion
# pixels per cell
PPC = 20
# cells per block
CPB = 2
# ORI
ORI = 9
# hist bins
HB = 6
# hist range
HR = (0, 256)
# downsample size
RSIZE = (10, 10)
# color space
CSPACE = "YCrCb"
# Channels to use
CHANS=[1,2,3]

# Define a function to compute binned color features

def bin_spatial(img):
    # Convert image to new color space (if specified)
    if CSPACE != 'BGR':
        image = cv2.cvtColor(img, CS[CSPACE])
    else:
        image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(image, RSIZE).ravel()
    # Return the feature vector
    return features


def color_hist(img):
    # Compute the histogram of the color channels separately
    channel_hist = []
    channel_hist.append(np.histogram(img[:, :, 0], bins=HB))
    channel_hist.append(np.histogram(img[:, :, 1], bins=HB))
    channel_hist.append(np.histogram(img[:, :, 2], bins=HB))
    # Concatenate the histograms into a single feature vector
    channel_hist = [chan[0]
                    for i, chan in enumerate(channel_hist) if i+1 in CHANS]
    hist_features = np.concatenate((channel_hist))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def gaussian_blur(img, kernel_size=3):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def hog_by_chan(img, rav=True, vis=False, gaus=5):
    """
    Compute individual channel HOG features for the image
    """
    img = cv2.cvtColor(img, CS[CSPACE]).astype(np.float32)
    # seperate the channels
    ch1 = img[:, :, 0]
    ch2 = img[:, :, 1]
    ch3 = img[:, :, 2]
    # if we are applying gaussian blur
    if gaus > 0:
        ch1 = gaussian_blur(ch1, gaus)
        ch2 = gaussian_blur(ch2, gaus)
        ch3 = gaussian_blur(ch3, gaus)
    # make a channel list
    channel = [ch1, ch2, ch3]
    hoglist = []
    imagelist = []
    # iterate through the channels that we are using
    for ch in CHANS:
        h = hog(channel[ch-1], orientations=ORI, pixels_per_cell=(PPC, PPC),
                cells_per_block=(CPB, CPB), visualise=vis,
                feature_vector=False, block_norm='L2-Hys')
        if not vis:
            hoglist.append(h)
        else:
            hoglist.append(h[0])
            imagelist.append(h[1])
    if rav:
        HOG = np.ravel(hoglist)
        if vis:
            return HOG, imagelist
        # if we aren't returning an image, just return the hoglist
        return HOG
    elif vis:
        return hoglist, imagelist
    return hoglist


def extract_features(image, Hogs=True):
    # Assure that all images are of the same size and type
    image = cv2.resize(image, (64, 64)).astype(np.float32)
    # Apply bin_spatial() to get spatial color features
    s_features = bin_spatial(image)
    # Apply color_hist() to get color histogram features
    hist_features = color_hist(image)
    if Hogs:
        # Get HOG features
        hog_features = hog_by_chan(image, rav=True)
        # Return feature vectors
        return np.concatenate((s_features, hist_features, hog_features))
    return np.concatenate((s_features, hist_features))


def norm_split_shuffle(car_features, not_car_features, labels):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float32)
    # Get a seperate Training and Test dataset
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.17, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    # Apply the scaler seperatly to test set, so the SVM doesn't "peak" at testing data
    X_test = X_scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_scaler


def retrieveImagePathList(folder_regx):
    # a helper function to get the images in a folder given a folder regx
    path_list = []
    for i in glob.glob(folder_regx, recursive=True):
        path_list.append(i)
    return path_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on image copy using cv2.rectangle()
    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thickness=thick)
    # return the image copy with boxes drawn
    return draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255, 0, 0), 6)
    # Return the image
    return img

def get_labeled_boxes(labels):
    boxes=[]
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        # and add append it to the boxes list
        boxes.append(((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy))))
    return boxes

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, decide, cells_per_step=5):
    # make a copy of the image
    img = img.astype(np.float32)
    boxes = []

    # Only searching relevant parts of the image
    imgToSearch = img[ystart:ystop, :, :]
    imshape = imgToSearch.shape

    # resize the image to the scale 1/n
    if scale != 1:
        imgToSearch = cv2.resize(imgToSearch, (np.int(
            imshape[1]/scale), np.int(imshape[0]/scale))).astype(np.float32)
        imshape = imgToSearch.shape

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    # how many blocks do we have per window
    nblocks_per_window = (window // PPC) - CPB + 1
    # blocks in the x axis
    nxblocks = (imgToSearch.shape[1] // PPC) - CPB + 1
    # blocks in the y axis
    nyblocks = (imgToSearch.shape[0] // PPC) - CPB + 1
    # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1 
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    # windows = slide_window(imgToSearch,xy_overlap=(0.25, 0.25))

    # get the image hog features for the whole image
    hog_features_list = hog_by_chan(imgToSearch,rav=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            hog_feat_list = []
            # get our position in the image
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # define the coords to start and stop getting the image patch
            x_y = [xpos*PPC, ypos*PPC]
            x1_y1 = [xpos*PPC+window, ypos*PPC+window]
            # Extract the image patch
            subimg = cv2.resize(
                imgToSearch[x_y[1]:x1_y1[1], x_y[0]:x1_y1[0]], (64, 64))
            # get the HOG features from the image patch
            for hf in hog_features_list:
                hog_feat_list.append(
                    hf[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
            hog_features = np.hstack(tuple(hog_feat_list))
            # Get color/spacial features
            s_and_hist_feats = extract_features(subimg, Hogs=False)
            # concatenate and transform features
            # Scale features
            test_features = X_scaler.transform(np.concatenate(
                (s_and_hist_feats, hog_features)).reshape(1, -1))
            # Make a prediction
            decision = svc.decision_function(test_features)
            # If the decision is above a threshold append to the output the resulting bounding box
            if decision > decide:
                xbox_left = np.int(x_y[0]*scale)
                ytop_draw = np.int(x_y[1]*scale)
                win_draw = np.int(64*scale)
                boxes.append([(xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw, ytop_draw+win_draw+ystart)])
    return boxes

# Define a class to keep track of found vehicles
import random
class vehicle_tracker:
    def __init__(self):
        self.n_frames = 15
        # Number of vehicles found
        self.nVehicles = 0
        # Heatmaps for vehicles
        self.vt_heatmap = []
        # bounding boxes
        self.bboxes = deque(maxlen=self.n_frames)
        self.count = 0
        # threshold for heatmap
        self.thresh = 2
        self.labels = []
        # Wanted to track individual vehicles
        # self.vehicles = []
        # self.vehIds = []
    
    def determine_thresh(self):
        if self.count < self.n_frames:
            self.count += 1
            if self.count//2 == 0:
                self.thresh =  1
            else:
                self.thresh =  self.count//2
        else:
            self.thresh = self.n_frames//2
    
    def gen_heat_map(self, current_hmap, labeled_bboxes):
        boxes = [i for i in labeled_bboxes]
        """ unused code to for potentially identifying indvidual vehicles and tracking them"""
        # reg = False
        # for i,box in enumerate(boxes):
        #     for v in self.vehicles:
        #         if v.registered(box):
        #             reg = True
        #             boxes[i] = (((v.boxY[0]+box[0][0])//2,(v.boxX[0]+box[0][1])//2),
        #                         ((v.boxY[1]+box[1][1])//2,(v.boxX[1]+box[1][1])//2))
        #             v.set_new_center(boxes[i])
        #             break
        #     if not reg:
        #         self.vehicles.append(vehicle(box))
        #         self.vehIds.append(self.vehicles[-1].id)
        # # de-register vehicles that are no longer detected
        # self.vehicles = [v for i,v in enumerate(self.vehicles) if v.id in self.vehIds]
        
        # add heat from previous frames to current heatmap
        add_heat(current_hmap,self.bboxes)
        current_hmap = apply_threshold(current_hmap, self.thresh)
        self.vt_heatmap = np.clip(current_hmap, 0, 255)
        # add the labels from the current heatmap to the list
        self.bboxes.extend(boxes)
        # calculate a new threshold
        self.determine_thresh()
        # create labels for the heatmap
        self.labels = label( self.vt_heatmap )

""" unused code to for potentially identifying indvidual vehicles and tracking them"""
# class vehicle:
#     def __init__(self, bbox):
#         self.center = (bbox[1][1]-bbox[0][1],bbox[1][0]-bbox[0][0]) #(x,y)
#         self.boxX = [bbox[0][1],bbox[1][1]]
#         self.boxY = [bbox[0][0],bbox[1][0]]
#         self.id = random.random()
    
#     def registered(self, bbox):
#         center = (bbox[1][1]-bbox[0][1],bbox[1][0]-bbox[0][0])
#         if abs(self.center[1]-center[1])+abs(self.center[0]-center[0]) < 64*2:
#             return True
#         else:
#             return False

#     def set_new_center(self,bbox):
#         self.center = (bbox[1][1]-bbox[0][1],bbox[1][0]-bbox[0][0]) #(x,y)
#         self.boxX = [bbox[0][1],bbox[1][1]]
#         self.boxY = [bbox[0][0],bbox[1][0]]
    
def detector(img, tracker, thresh=1, search_area=None, decide=0, cells_per_step=2):
    """ Main function for detecting vehicles and drawing bounding boxes around them"""
    image = np.copy(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    carmap = []
    # define a default search if one isn't given
    if not search_area:
        search_area = [[370, 626, 1.5],[370,370+64*4,2]]
    for sa in search_area:
        carmap.extend(find_cars(image, sa[0], sa[1], sa[2],
                        clf, X_scaler, decide, cells_per_step=cells_per_step))
    heat = np.zeros_like(image[:, :, 0]).astype(np.float32)
    # Add heat to each box in box list
    add_heat(heat, carmap)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, thresh)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find boxes from heatmap using label function
    labels = label( heatmap )
    # pass the labels to the tracker to generate a multi-frame averaged heatmap
    tracker.gen_heat_map(heat,get_labeled_boxes(labels))
    # Create a small heatmap image to put in the corner of the video
    hmap = np.zeros_like(image)
    hmap[:, :, 0] = tracker.vt_heatmap
    hmap[:, :, 1] = tracker.vt_heatmap
    hmap[:, :, 2] = tracker.vt_heatmap*(255//np.amax(tracker.vt_heatmap))
    wshp = image.shape
    smallhmap = cv2.resize(hmap, (np.int(wshp[1]/4), np.int(wshp[0]/4)))
    sshp = smallhmap.shape
    # redraw the image with bounding boxes and a heatmap
    window_img = draw_labeled_bboxes(image,tracker.labels )
    window_img[0:sshp[0], wshp[1]-sshp[1]:wshp[1]] = smallhmap
    return cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    # Note: all images are expected to be in the BGR color space when passed to a function
    carFiles = retrieveImagePathList('dataset/vehicles/**/*.png')
    notCarFiles = retrieveImagePathList('dataset/non-vehicles/**/*.png')
    carFeatures = []
    notCarFeatures = []
    # Option to take a fraction of the dataset 
    sample_size = 0
    fraction_of_car_dataset = 1 #len(carFiles)//sample_size
    fraction_of_not_car_dataset = 1 #len(notCarFiles)//sample_size

    # Print statements act as comments for this section
    if os.path.exists("pantry/features.p"):
        print("Opening pickeld features.  O_o vinegary\n")
        features = pickle.load(open("pantry/features.p", "rb"))
        carFeatures = features["car"]
        notCarFeatures = features["notCar"]
        labels = features["labels"]
    else:
        print("Extracting features from image files")
        t = time.time()
        # get the vehicle and non vehicle features
        for i, file in enumerate(carFiles):
            if not(i % fraction_of_car_dataset):
                carImg = cv2.imread(file)
                carFeatures.append(extract_features(carImg))
        for i, file in enumerate(notCarFiles):
            if not(i % fraction_of_not_car_dataset):
                notCarImg = cv2.imread(file)
                notCarFeatures.append(extract_features(notCarImg))
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract features...')
        print("feature length: ", len(carFeatures[0]))
        # pickle them for later consumption
        features = {}
        features["car"] = carFeatures
        features["notCar"] = notCarFeatures
        print("Generating Labels")
        labels = np.hstack(
            (np.ones(len(carFeatures)), np.zeros(len(notCarFeatures))))
        features["labels"] = labels
        pickle.dump(features, open("pantry/features.p", "wb"))
        print("features pickled and put in pantry/")

    if os.path.exists("pantry/classifier.p"):
        print("Opening pickled classifier, Don't you love the sound of popping open a new jar?")
        classifier = pickle.load(open("pantry/classifier.p", "rb"))
        clf = classifier['classifier']
        if os.path.exists("pantry/scaler.p"):
            print("Opening pickled scaler, it doesn't spoil if you seal it right.")
            scaler = pickle.load(open("pantry/scaler.p", "rb"))
            X_scaler = scaler['X_scaler']
        else:
            t = time.time()
            print("Nomalizing, spliting and shuffling data")
            X_train, X_test, y_train, y_test, X_scaler = norm_split_shuffle(
                carFeatures, notCarFeatures, labels)
            t2 = time.time()
            pickle.dump({"X_scaler": X_scaler}, open("pantry/scaler.p", "wb"))
            print("Scaler  pickled and put in pantry/")
    else:
        t = time.time()
        print("Nomalizing, spliting and shuffling data")
        X_train, X_test, y_train, y_test, X_scaler = norm_split_shuffle(
            carFeatures, notCarFeatures, labels)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to normalize, split and shuffle data...')
        pickle.dump({"X_scaler": X_scaler}, open("pantry/scaler.p", "wb"))
        print("Scaler  pickled and put in pantry/")
        print("Fitting linear classifier to dataset")
        t = time.time()
        clf = LinearSVC().fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("Accuracy of classifier = ", score)
        # pickle them for later consumption
        pickle.dump({"classifier": clf}, open("pantry/classifier.p", "wb"))
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train, fit and save classifier...')
        print("Classifier pickled and put in pantry/")

    # define a threshold, decision threshold, search area and init the tracker
    thresh = 1
    decide = 1.5
    search = [[400,464,1],[370, 500, 1.13],[390,600,2],[370,700,3]]
    tracker = vehicle_tracker()
    
    # Testing on images!!!!
    # test_img_files = retrieveImagePathList('test/mid*.png')
    # test_img_files = retrieveImagePathList('test/close*.png')
    # test_img = []
    # for file in test_img_files:
    #     test_img.append(cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB))

    # imout = []
    # for img in test_img:
    #     imout.append(detector(img,  tracker, search_area=search, 
    #                 thresh=thresh, decide=decide, cells_per_step=5))

    # plt.figure(figsize=(20, 20))
    # sp = 1
    # for d, i in enumerate(imout):
    #     plt.subplot(3,4, sp)
    #     plt.imshow(i)
    #     plt.title("image %i" % d)
    #     sp += 1
    # plt.show()

    outvid = 'output_videos/proj_out_filter_0.mp4'
    clip1 = VideoFileClip('video/project_video.mp4', audio=False)
    result_clip = clip1.fl_image (lambda x: detector(x, tracker, search_area=search, thresh=thresh,  decide=decide)) #NOTE: this function expects color images!!
    result_clip.write_videofile(outvid, audio=False, verbose=False)
