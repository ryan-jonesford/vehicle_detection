import os
import glob
import pickle
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.ndimage.measurements import label
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
import time
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


# Define a color space dict. All image reads done by cv2.imread()
cs = { 'HSV':    cv2.COLOR_BGR2HSV, 
       'YUV':    cv2.COLOR_BGR2YUV, 
       'YCrCb':  cv2.COLOR_BGR2YCrCb,
       'LUV':    cv2.COLOR_BGR2LUV,
       'HLS':    cv2.COLOR_BGR2HLS }
# Declare these globally, so there isn't confusion
# pixels per cell
ppc = 8 
# cells per block
cpb = 1
# orient
ori = 8
# hist bins
hb = 30
# hist range
hr = (0, 256)
# downsample size
rsize = (30,30)
# color space
cspace = "LUV"

# Define a function to compute binned color features  
def bin_spatial(img, color_space=cspace, size=rsize):
    # Convert image to new color space (if specified)
    if color_space != 'BGR': image = cv2.cvtColor(img,cs[color_space])
    else: image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(image,size).ravel()
    # Return the feature vector
    return features
    
# def bin_spatial(img, size=(32, 32)):
#     color1 = cv2.resize(img[:,:,0], size).ravel()
#     color2 = cv2.resize(img[:,:,1], size).ravel()
#     color3 = cv2.resize(img[:,:,2], size).ravel()
#     return np.hstack((color1, color2, color3))

def color_hist(img, nbins=hb, bins_range=hr):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def hog_by_chan(img,pix_per_cell=ppc,cell_per_block=cpb,orient=ori, rav=True, 
                c_space='LUV', chans=[1,2,3], vis=False):
    """
    Compute individual channel HOG features for the image
    """
    img = cv2.cvtColor(img, cs[c_space]).astype(np.float32)
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    hoglist=[]
    imagelist=[]
    if 1 in chans:
        h = hog(ch1,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                        cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
                        feature_vector=False,block_norm='L2-Hys')
        if not vis:
            hoglist.append(h)
        else:
            hoglist.append(h[0])    
            imagelist.append(h[1])
    if 2 in chans:
        h = hog(ch1,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                        cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
                        feature_vector=False,block_norm='L2-Hys')
        if not vis:
            hoglist.append(h)
        else:
            hoglist.append(h[0])    
            imagelist.append(h[1])
    if 3 in chans:
        h = hog(ch1,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                        cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
                        feature_vector=False,block_norm='L2-Hys')
        if not vis:
            hoglist.append(h)
        else:
            hoglist.append(h[0])    
            imagelist.append(h[1])
    if rav:
        HOG = np.ravel(hoglist)
        if vis:
            return HOG, imagelist
        return HOG
    elif vis:
        return hoglist,imagelist
    return hoglist

def extract_features(image, cspace=cspace, spatial_size=rsize,
                        hist_bins=hb, hist_range=hr, Hogs=True):
    # Assure that all images are of the same size and type
    image = cv2.resize(image,(64,64)).astype(np.float32)
    # Apply bin_spatial() to get spatial color features
    s_features = bin_spatial(image, size=spatial_size)
    # Apply color_hist() to get color histogram features
    hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
    if Hogs:
        # Apply get_hog_features to get HOG features
        hog_features = hog_by_chan(image,rav=True)
        # Return feature vectors
        return np.concatenate((s_features,hist_features,hog_features))
    return np.concatenate((s_features,hist_features))

def norm_split_shuffle(car_features, not_car_features, labels):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, not_car_features)).astype(np.float32)                        
    # Get a seperate Training and Test dataset
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X,labels,test_size=0.17,random_state=rand_state)
   
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_scaler

def retrieveImagePathList(folder_regx):
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

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.25, 0.25)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None or y_start_stop[0] == None:
        x_start_stop=[0,img.shape[1]]
        y_start_stop=[0,img.shape[0]]
    # Compute the span of the region to be searched
    span = (x_start_stop[1]-x_start_stop[0], y_start_stop[1]-y_start_stop[0])
    # Compute the number of pixels per step in x/y
    pix_per_step = ((xy_window[0]*xy_overlap[0]),(xy_window[1]*xy_overlap[1]))
    # Compute the number of windows in x/y
    windows = (1+(span[0]-xy_window[0])//pix_per_step[0],1+(span[1]-xy_window[1])//pix_per_step[1])
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for xi in range(int(windows[0])):
        for yi in range(int(windows[1])):
            # Calculate each window position
            xy1 = (int(x_start_stop[0]+xi*pix_per_step[0]),int(y_start_stop[0]+yi*pix_per_step[1]))
            xy2 = (int(xy1[0]+xy_window[0]),int(xy1[1]+xy_window[1]))
            window_list.append((xy1,xy2))
        # Append window position to list
    # Return the list of windows
    return window_list

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
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler):   
    draw_img = np.copy(img)
    img = img.astype(np.float32)
    boxes = []
    
    # Only searching relevant parts of the image
    imgToSearch = img[ystart:ystop,:,:]
    imshape = imgToSearch.shape

    if scale != 1:
        imgToSearch = cv2.resize(imgToSearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale))).astype(np.float32)
        imshape = imgToSearch.shape
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // ppc) - cpb + 1
    cells_per_step = 6  # Instead of overlap, define how many cells to step
    nxblocks = (imgToSearch.shape[1] // ppc) - cpb + 1
    nyblocks = (imgToSearch.shape[0] // ppc) - cpb + 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    # windows = slide_window(imgToSearch,xy_overlap=(0.25, 0.25))

    # get the image hog features
    hog_features_list = hog_by_chan(imgToSearch, rav=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            hog_feat_list = []
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            x_y = [xpos*ppc,ypos*ppc]
            x1_y1 = [xpos*ppc+window,ypos*ppc+window]
            # Extract the image patch
            subimg = cv2.resize(imgToSearch[x_y[1]:x1_y1[1], x_y[0]:x1_y1[0]], (64,64))
            for hf in hog_features_list:
                hog_feat_list.append(hf[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
            hog_features = np.hstack(tuple(hog_feat_list))
            # Get color features
            s_and_hist_feats = extract_features(subimg,Hogs=False, spatial_size=rsize)
            # concatenate and transform features
            test_features = X_scaler.transform(np.concatenate((s_and_hist_feats, hog_features)).reshape(1, -1))   
            # Scale features and make a prediction
            test_prediction = svc.predict(test_features)
            # Define boxes around positive predictions
            if test_prediction == 1:
                xbox_left = np.int(x_y[0]*scale)
                ytop_draw = np.int(x_y[1]*scale)
                win_draw = np.int(64*scale)
                boxes.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])
    return boxes

# Define a class to keep track of found vehicles
class vehicle_tracker():
    def __init__(self):
        # Number of vehicles found
        self.nVehicles = 0
        # Bounding boxes for vehicles
        self.vBBoxes = []

def detector(img, tracker, thresh=1, search_area=None):
    image = np.copy(img)
    carmap=[]
    if not search_area:
        search_area = [[375,500,.5],[375,525,1],[475,675,2.5],[500,720,3]]
    for sa in search_area:
        carmap =carmap+find_cars(image, sa[0], sa[1], sa[2], clf, X_scaler)
    heat = np.zeros_like(image[:,:,0]).astype(np.float32)
    # Add heat to each box in box list
    add_heat(heat,carmap)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,thresh)
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # Create a small heatmap image to put in the corner of the video
    hmap = np.zeros_like(image)
    hmap[:,:,0] = heatmap*(255//np.amax(heatmap))
    hmap[:,:,1] = heatmap
    hmap[:,:,2] = heatmap
    wshp=image.shape
    smallhmap = cv2.resize(hmap,(np.int(wshp[1]/4), np.int(wshp[0]/4)))
    sshp=smallhmap.shape
    window_img = draw_labeled_bboxes(image, labels)
    window_img[0:sshp[0], wshp[1]-sshp[1]:wshp[1]] = smallhmap
    return window_img

if __name__ == '__main__':
    # Note: all images are expected to be in the BGR color space when passed to a function
    carFiles = retrieveImagePathList('vehicles/vehicles/**/*.png')
    notCarFiles = retrieveImagePathList('non-vehicles/non-vehicles/**/*.png')
    carFeatures = []
    notCarFeatures = []

    if os.path.exists("pantry/features.p"):
        print("Opening pickeld features.  O_o vinegary\n")
        features = pickle.load( open( "pantry/features.p", "rb" ) )
        carFeatures = features["car"]
        notCarFeatures = features["notCar"]
        labels = features["labels"]
    else:
        print("Extracting features from image files")
        t=time.time()
        # get the vehicle and non vehicle features
        for file in carFiles:
            carImg = cv2.imread(file)
            carFeatures.append(extract_features(carImg))
        for file in notCarFiles:
            notCarImg = cv2.imread(file)
            notCarFeatures.append(extract_features(notCarImg))
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract features...')
        # pickle them for later consumption
        features = {}
        features["car"] = carFeatures
        features["notCar"] = notCarFeatures
        print("Generating Labels")
        labels = np.hstack((np.ones(len(carFeatures)), np.zeros(len(notCarFeatures))))
        features["labels"] = labels
        pickle.dump( features, open( "pantry/features.p", "wb" ) )
        print("features pickled and put in pantry/")
    
    t=time.time()
    print("Nomalizing, spliting and shuffling data")
    X_train, X_test, y_train, y_test, X_scaler = norm_split_shuffle(carFeatures,notCarFeatures,labels)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to normalize, split and shuffle data...')

    grid = None
    if os.path.exists("pantry/paramVals.p"):
        print("Opening pickeld paramValues... They don't spoil if sealed right :)\n")
        paramVals = pickle.load( open( "pantry/paramVals.p", "rb" ) )
        C = paramVals["classify__C"]
        gamma = paramVals["classify__gamma"]
        kernel = paramVals["classify__kernel"]
    else:
        t = time.time()
        # source: http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
        pipe = Pipeline([
            ('reduce_dim', PCA()),
            ('classify', SVC())
        ])
        #http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py
        # source: http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
        # go through many C and gamma variables to get the ones that should work the best
        # C_range = np.logspace(-1, 4, 5)
        C_range = [10,100,1000]
        gamma_range = [1e-4]
        n_components = [16,64]
        param_grid = [
            {
                'reduce_dim__n_components': n_components,
                'classify__kernel':['linear'],
                'classify__C': C_range,
                'classify__gamma': gamma_range,
            }
        ]
        # parameters = { 'kernel':('linear', 'poly', 'rbf'), 'C': C_range, 'gamma':gamma_range}
        print("Starting Grid Search for best parameters")
        # grid = GridSearchCV(SVC(), parameters, n_jobs=6, verbose=5)
        grid = GridSearchCV(pipe, cv=3, n_jobs=6, verbose=5, param_grid=param_grid)
        # fit to a subsample of the data, this takes a long time as it is and we are only looking
        # for a rough estimate of which hyper paramters will work the best
        grid.fit(X_train[:1000], y_train[:1000])
        C = grid.best_params_["classify__C"]
        gamma = grid.best_params_["classify__gamma"]
        kernel = grid.best_params_["classify__kernel"]
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train and fit SVM...')
        # pickle them for later consumption
        pickle.dump( grid.best_params_, open("pantry/paramVals.p", "wb"))
        pickle.dump( grid.best_estimator_, open("pantry/estimator.p", "wb"))
        print("The best parameters are %s with a score of %0.2f"
            % (grid.best_params_, grid.best_score_))
        print("\nParameters pickled and put in pantry/")
            
    
    if os.path.exists("pantry/classifier.p"):
        print("Opening pickled classifier, Don't you love the sound of popping open a new jar?")
        classifier = pickle.load( open( "pantry/classifier.p", "rb" ) )
        clf = classifier['classifier']
        # uncomment the next line to get an accuracy readout of the read in classifier
        # print('This Classifier has a Test Accuracy of ', round(clf.score(X_test, y_test), 4)*100,"%")
    else:
        print("Fitting best classifier to full dataset")
        t = time.time()
        svm = SVC()
        pca = PCA()
        pipe = Pipeline([('reduce_dim', pca), ('classify', svm)])
        clf = pipe.set_params(classify__kernel=kernel, classify__C=C,classify__gamma=gamma, classify__probability=True).fit(X_train, y_train)
        pickle.dump( clf, open("pantry/classifier.p", "wb"))
        score = clf.score(X_test,y_test)
        print("Accuracy of classifier = ", score)
        # pickle them for later consumption
        pickle.dump( {"classifier":clf }, open("pantry/classifier.p", "wb"))
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train, fit and save classifier...')
        print("Classifier pickled and put in pantry/")

    tracker = vehicle_tracker()
    image = cv2.imread("test/tester003.png")
    tracker = vehicle_tracker()
    imout = detector(image, tracker)
    plt.figure(figsize=(12,10))
    plt.imshow(cv2.cvtColor(imout,cv2.COLOR_BGR2RGB))
    plt.show()
    # outvid = 'output_videos\\test.mp4'
    # clip1 = VideoFileClip('video\\hard.mp4', audio=False)
    # result_clip = clip1.fl_image (lambda x: detector(x, tracker)) #NOTE: this function expects color images!!
    # result_clip.write_videofile(outvid, audio=False, verbose=False)
