# Vehicle Driving Assistant

# Problem Statement
The project aims to provide a comprehensive set of assistance features to aid the driver (or autonomous vechicle) to drive safely. This includes a number of indicators 
about the environment, the major cue being detection of potholes in the road ahead.

# Motivation
Autonomous vehicles has been a common term in our day to day life with car manufacturers like Tesla shipping cars that are SAE Level 3. While these vehicles include
a slew of features such as parking assistance and cruise control, they've mostly been tailored to foreign roads. Potholes, and the abundance of them, is something that is
unique to our Indian roads. We believe that successful detection of potholes from visual images can be applied in a variety of scenarios.
Moreover, the sheer variety in the color, shape and size of potholes makes this problem an apt candidate to be solved using modern machine learning techniques. 

# Related Works
Mention related papers - what they've done. What they've not done.


# Progress so far (Dataset, description of approach(features, modeling))

We started out with the naive approach - try to detect the potholes directly from pictures using image processing techniques. We got a collected a small dataset from Google
images and applied classic techniques like edge and contour detection. The problem with Google images of potholes is that they're heavily biased - the potholes are 
shown in such a way that it is the center of attraction in the picture, with good lighting and clear demarcation from the environment. While our technique worked for a 
limited subset, the results were far from satisfactory. We got too much noise and too little usable data to go on. Detecting potholes, which may be large or small, 
circular or quadrilateral, filled with liquid or dust and overall varied in nature, was not a straightforward problem.

In order to get dataset that is rooted in the real world, we went out and recorded 360p video of the roads in and around Electronic City phase 2. We managed to get about 20
minutes of footage of varying terrains with diverse features that are commonly encountered on Indian roads. The footage was preprocessed to reduce jitter induced in the
video due to it being recorded by the pillion rider of a moving motorcycle. 
Collected data: YOUTUBE LINK

We expected the detection of potholes to be simplified if a localized area in the image can be found where the chance of occurrence of the pothole is relatively high. In
essence, we decided to extract the road from the surrounding environment.

In order to remove the external environment and extract the road from the given frame, we tried a variety of techniques. 

* Choose a channel
    We had to come up with an input representation where the difference between the road and the surrounding was accentuatted. We tried Grayscale images but the contrast
    wasn't enough to facilitate extraction. Much better results were found by converting the frame to HSV colorspace. We tried each channel in isolation and got satisfactory results with the S and V channels. 
* Remove Noise
    Now that we had a channel to work on, wee had to reduce the amount of noise in the frame. Between Gaussian Blur and Median Blur, we got better results with Median Blur 
    with a relatively large kernel size to reduce the noise inherent in the picture. 
* Threshold to get a suitable mask
    Now that we had a denoised version of the channel with maximum variance between the road and surrounding, we applied Otzu's thresholding to extract the part of the image
    containing the road. We got a relatively good output but there was still noise in the thresholded mask.
* Clean up the mask
    (EXPLAIN MORPHOLOGICAL TRANSFORMATION)There were still stray islands in the mask, i.e small localized black(white) regions within the white(black) areas. In order to even these out, we used the following two
    morphological transformation methods in succession -
    ** Closing : It is dialation followed by erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.
    ** Opening : It is erosion followed by dialation. It is useful in removing noise from the black areas.
    We used a relatively large kernel for both the operations to get a fairly clean mask.
* Application of mask
    The obtained mask was applied on the original image and the road was extracted out. 

    Mask applied on collected dataset: YOUTUBE LINK.

# Future Work
EXPAND on these  - 
* feature extraction
* Application of standard ML algorithms
* Comparitive analysis of obtained results.
