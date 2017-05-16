
# Vehicle Driving Assistant

### Code
The code has been organized as follows:
* [preprocess.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/preprocess.py)
* [pothole_ml.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/pothole_ml.py)
* [pothole_img_proc.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/pothole_img_proc.py)


#### [preprocess.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/preprocess.py)

Uses image processing to isolate and extract the road. On a high-level, it follows the given steps:
* Convert to HSV colorspace.
* Sample a road area.
* Threshold based on sample patch to get image mask.
* Apply contour detection on the obtained mask.
* Calculate the convex hull of the largest contour. 
* Extract region of interest from input based on calculated hull mask.

```extract_road``` method returns the extracted road area.

#### [pothole_ml.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/pothole_ml.py)

Discrimative features are extracted from the isolated road obtained from ```extract_road```. 

Feature extraction is done in two ways:
* Downsampling the original image and using the raw pixel values as features
* Create a colorhistogram of the input image as features

The extracted features are fed into a classifier and the performance is recorded. 

The following models have been used for classification:
* Naive Bayes
* K-Nearest Neighbour
* SVM
* Decision Trees
* AdaBoost (using Decision Trees)

#### [pothole_img_proc.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/pothole_img_proc.py)
Contains a bunch of unsuccessful attempts at pothole isolation using blob detection and contour dilation. 

### Data

Dataset used by [pothole_ml.py](https://github.com/crunchbang/MP_Project/blob/master/ML/final_code/pothole_ml.py): 
https://drive.google.com/open?id=0B6KoReKNTX1cQ2FnYmNha24tZGs

Dataset with negative and positive examples separated: https://drive.google.com/drive/folders/0B7LHCitTUdEYZFEwNWo4V2RldjQ?usp=sharing

-----
 Comparitive results between different feature extraction methods and classification models have been mentioned in the report.



