Vehicle Driving Assistant
-------------------------

## Brief Description
This project aims to provide a comprehensive set of assistance features to aid the driver (or autonomous vehicle) to drive safely. This includes a number of indicator and cues regarding the environment.  

### Problem FormulationApplication/System (Driving Assistance)	MP Module (Obstacle detection)		ML task (Multiclass Classification)			Features, Models, Optimization algorithm (To be decided based on empirical observations)				

## Dataset 
360p video of roads in and around Electronic City Phase was recorded by us. We used a Cannon 1300D to record the video. 

## Proposed Plan of Execution:

### Phase 1 (End of March) 

* Data collection and video pre-processing

Video of roads in Electronic City was collected manually with a DSLR camera held by a pillion rider on a moving motorcycle. About 30 minutes worth of video was collected for Phase 1 of the project. The video was pre-processed to reduce excessive jitter.

### Phase 2 

* Pothole detection in video frames.

### Phase 3

* Improving accuracy of the current model.
* Comparing the result with alternate models.

### Phase 4

* Detection of other obstacles (pedestrians, oncoming traffic etc).
* Provide driving cues based on detected entities. 


## Main Challenges:

* Procuring large and varied data to work on.

* Presence of noise, shadows and excessive jitter in data

* Diversity and non-uniformity among roads and potholes which makes it difficult to generalise. 

* Detecting obstacles with an obstructed field of view.

* Consistency in object detection with increase in speeds.

## Learning Objectives

* Finding a real world problem and formulating it as a machine perception problem.

* Get acquainted with different aspects of image analysis for feature extraction.

* Be able to extract appropriate features and apply machine learning algorithms to solve the problem.

* Be able to read current research papers and understand the issues discussed.

* Obtaining and cleaning data.
