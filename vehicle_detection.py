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
import time

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
        h1 = hog(ch1,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                        cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
                        feature_vector=False,block_norm='L2-Hys')
        hoglist.append(h1[0])
        if vis:
            imagelist.append(h1[1])
    if 2 in chans:
        h2 = hog(ch2,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                    cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
                    feature_vector=False,block_norm='L2-Hys')
        hoglist.append(h2[0])
        if vis:
            imagelist.append(h2[1])
    if 3 in chans:
        h3 = hog(ch3,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
                    cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
                    feature_vector=False,block_norm='L2-Hys')
        hoglist.append(h3[0])
        if vis:
            imagelist.append(h3[1])
    if rav:
        hlist = []
        for HOG in hoglist:
            hlist.append(HOG.ravel())
        if vis:
            return np.hstack(tuple(hlist)), imagelist
        return np.hstack(tuple(hlist))
    elif vis:
        return hoglist,imagelist
    return hoglist
 
# def extract_features(imageFiles, cspace=cspace, spatial_size=rsize,
#                         hist_bins=hb, hist_range=hr):
#     features = []
#     for file in imageFiles:
#         # Assure that all images are of the same size and type
#         image = cv2.resize(cv2.imread(file),(64,64)).astype(np.float32)
#         # Apply bin_spatial() to get spatial color features
#         s_features = bin_spatial(image, size=spatial_size)
#         # Apply color_hist() to get color histogram features
#         hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
#         # Apply get_hog_features to get HOG features
#         hog_features = hog_by_chan(image,feature_vec=True)
#         # Append the new feature vector to the features list
#         features.append(np.concatenate((s_features,hist_features,hog_features)))
#     # Return list of feature vectors
#     return features

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

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler):   
    draw_img = np.copy(img)
    img = img.astype(np.float32)
    boxes = []
    
    # Only searching relevant parts of the image
    imgToSearch = img[ystart:ystop,:,:]
    imshape = imgToSearch.shape

    if scale != 1:
        imgToSearch = cv2.resize(imgToSearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        imshape = imgToSearch.shape
    
    windows = slide_window(imgToSearch)
    
    # Compute individual channel HOG features for the entire (cropped) image
    # HOG = hog_by_chan(imgToSearch, rav=False)
    # hogger = get_hog_features(imgToSearch, feature_vec=False)
    
    for window in windows:
        # Extract HOG for this patch
        x_y = window[0]
        x1_y1 = window[1]
        # hog_feat = []
        # for h in HOG:
        #     hog_feat = h[xpos[0]:ypos[0], xpos[1]:ypos[1]].ravel() 
        # hog_features = np.hstack(tuple(hog_feat))
        # Extract the image patch
        subimg = cv2.resize(imgToSearch[x_y[1]:x1_y1[1], x_y[0]:x1_y1[0]], (64,64))
        hog_features = hog_by_chan(subimg, rav=True)
        s_and_hist_feats = extract_features(subimg,Hogs=False)
        # Get color features
        # spatial_features = bin_spatial(subimg, size=spatial_size)
        # hist_features = color_hist(subimg)
        test_features = X_scaler.transform(np.concatenate((s_and_hist_feats, hog_features)).reshape(1, -1))   

        # Scale features and make a prediction
        test_prediction = svc.predict(test_features)
        
        if test_prediction == 1:
            xbox_left = np.int(x_y[0]*scale)
            ytop_draw = np.int(x_y[1]*scale)
            win_draw = np.int(64*scale)
            boxes.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])

    draw_img = draw_boxes(draw_img, boxes)                
    return draw_img

# Define a single function that can extract features using hog sub-sampling and make predictions
# def find_cars(img, ystart, ystop, scale, svc, X_scaler):   
#     draw_img = np.copy(img)
#     img = img.astype(np.float32)
#     boxes = []
    
#     # Only searching relevant parts of the image
#     imgToSearch = img[ystart:ystop,:,:]
#     imshape = imgToSearch.shape

#     if scale != 1:
#         imgToSearch = cv2.resize(imgToSearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
#         imshape = imgToSearch.shape

#     # Define blocks and steps as above
#     nxblocks = (imshape[1] // ppc) - cpb + 1
#     nyblocks = (imshape[0] // ppc) - cpb + 1 
    
#     # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#     window = 64
#     nblocks_per_window = (window // ppc) - cpb + 1
#     cells_per_step = 4  # Instead of overlap, define how many cells to step
#     nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
#     nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
#     if nysteps == 0: nysteps = 1
    
#     # Compute individual channel HOG features for the entire (cropped) image
#     HOG = hog_by_chan(imgToSearch, rav=False)
#     # hogger = get_hog_features(imgToSearch, feature_vec=False)
    
#     for xb in range(nxsteps):
#         for yb in range(nysteps):
#             ypos = yb*cells_per_step
#             xpos = xb*cells_per_step
#             # Extract HOG for this patch
#             hog_feat = []
#             for h in HOG:
#                 hog_feat = h[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#             hog_features = np.hstack(tuple(hog_feat))
#             # hogger_feat = hogger[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

#             xleft = xpos*ppc
#             ytop = ypos*ppc

#             # Extract the image patch
#             subimg = cv2.resize(imgToSearch[ytop:ytop+window, xleft:xleft+window], (64,64))
#             s_and_hist_feats = extract_features(subimg,Hogs=False)

#             # Get color features
#             # spatial_features = bin_spatial(subimg, size=spatial_size)
#             # hist_features = color_hist(subimg)
#             test_features = X_scaler.transform(np.concatenate((s_and_hist_feats, hog_features)).reshape(1, -1))   

#             # Scale features and make a prediction
#             test_prediction = svc.predict(test_features)
            
#             if test_prediction == 1:
#                 xbox_left = np.int(xleft*scale)
#                 ytop_draw = np.int(ytop*scale)
#                 win_draw = np.int(window*scale)
#                 boxes.append([(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)])

#     draw_boxes(draw_img, boxes)                
#     return draw_img

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

    if os.path.exists("pantry/paramVals.p"):
        print("Opening pickeld paramValues... They don't spoil if sealed right :)\n")
        paramVals = pickle.load( open( "pantry/paramVals.p", "rb" ) )
        C = paramVals["C"]
        gamma = paramVals["gamma"]
    else:
        t = time.time()
        # source: http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
        # go through many C and gamma variables to get the ones that should work the best
        C_range = np.logspace(-3, 3, 10)
        gamma_range = np.logspace(-3, 3)
        parameters = { 'kernel':('linear', 'poly', 'rbf'), 'C': C_range, 'gamma':gamma_range}
        print("Starting Grid Search for best parameters")
        grid = GridSearchCV(SVC(), parameters, verbose=5)
        # fit to a subsample of the data, this takes a long time as it is and we are only looking
        # for a rough estimate of which hyper paramters will work the best
        grid.fit(X_train, y_train)
        C = grid.best_params_["C"]
        gamma = grid.best_params_["gamma"]
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train and fit SVM...')
        # pickle them for later consumption
        pickle.dump( grid.best_params_, open("pantry/paramVals.p", "wb"))
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
        t = time.time()
        print("Fitting classifier")
        clf = SVC(kernel='linear', C=C, gamma=gamma, verbose=True)
        clf.fit(X_train, y_train)
        print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4)*100,"%")
        # pickle them for later consumption
        pickle.dump( {"classifier":clf }, open("pantry/classifier.p", "wb"))
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train, fit and save classifier...')
        print("Classifier pickled and put in pantry/")

    image = cv2.imread("test/example-image.png")
    window_img = find_cars(image, 440,656,1,clf,X_scaler)
    # windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[0,image.shape[0]], 
    #                 xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    # window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg                    
    plt.figure()
    plt.imshow(window_img)
    plt.show()
