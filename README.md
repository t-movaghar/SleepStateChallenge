# Detect Sleep States Challenge
This repository holds my attempt at the Child Mind Institute: Detect Sleep States Kaggle Challenge
https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states

## Overview

  The ultimate goal of my challenge attempt was to develop a model, trained on the wrist-worn accelerometer data provided, to accurately detect periods of sleep onset/wake.
  
  Performance was tenative, with validation loss significantly above training. However, after completing this project, I feel I have gained the experience necessary to tackle a similar, albeit less ambitious, problem with greater success.

### Data

* Data:
  * Type: parquet, csv
    * Input: Two Parquet files (training, testing) containing continuous accelerometer data recordings
    * Input: CSV file containing event (signal) labels for recordings in the training data. False onset/wake instances were left unlabeled.
  * Size: All files total to approximately 986.46 MB
  * Test set contains data for 200 patients
 
#### Preprocessing / Clean up

Data was significantly downsampled due to hardware limitations. 
Only observations from every 10th unique series_ID (lengths of patient data) was considered in the final model.

#### Data Visualization

Patient 038441c925bb Sleep Onset/Wake
![image](https://github.com/t-movaghar/SleepStateChallenge/assets/123412483/149cb51f-5be0-444a-9066-2b90a25b4a57)
Periods of abnormally low activity were unlabeled, as these periods do not signal an onset/wake event, but a period of time where the accelerometer was removed.
This is reflected in the visualization.

Patient ece2561f07e9 Sleep Onset/Wake
![image](https://github.com/t-movaghar/SleepStateChallenge/assets/123412483/2fd06559-7535-4ac6-a2af-55aac602e48e)

Patient cf13ed7e457a Sleep Onset/Wake
![image](https://github.com/t-movaghar/SleepStateChallenge/assets/123412483/f4dd33cd-a79d-45b7-b0d3-c62705ccfe49)

Data varies wildly between patients.

### Problem Formulation

  Features extracted from the accelerometer data (mean, max, and variance) were used as the input to my model. For supervised training, corresponding signals (onset/wake) would also be considered.
  The output was, ideally, a predicted set of signals corresponding to the features extracted from my testing dataset based on the associations made during training.
  The first neural network I attempted was composed of several dense layers, with dropout layers added to avoid overfitting and hidden layers employing ReLU activation. The output layer used the sigmoid activation function, and the model was compiled using the Adam optimizer and binary cross-entropy loss. Training was conducted with a batch size of 2048 for 10 epochs.
  The second neural network I attempted was composed of several dense layers employing ReLU activation. The output layer used the sigmoid activation function, and the model was compiled using the rmsprop optimizer and binary cross-entropy loss.

### Training

  Training was attempted using Tensorflow and Keras for model building.
  The use of a GPU for training was not possible.
  
  Training times varied depending on the number of epochs and batch_size specified, though training times typically fell between twenty and five minutes, even with the downsampled data. 
  Training on a CPU also exacerbated this issue. To combat this, batch size was increased from an initial, arbitrary value of 100 to 2048.

  Model 1
  ![image](https://github.com/t-movaghar/SleepStateChallenge/assets/123412483/041eb593-1330-42d1-b848-9b4eab160fcd)

  Model 2
  ![image](https://github.com/t-movaghar/SleepStateChallenge/assets/123412483/a7455e8d-d152-4c41-b618-a2557e459f40)


  Training was stopped arbitrarily.

  The features derived from the train_series.parquet were split into testing and validation datasets, with witch the training/validation loss was derived.

### Performance Comparison

  Accuracy metric consistently reached 0.9999 over every model, which is surprisingly high.
  This leads me to believe that some overfitting of my models may have taken place.
  
### Conclusions

  Improvements in training/validation loss was seen when the model was compiled using the rmsprop optimizer.

### Future Work

Sleep contributes to the maintinence of a healthy mood, body weight, and functioning immune system, to name a few. As such, finding ways to accurately predict sleep onset/wakeup times has a wide range of implications for future study and would help us gain valuable insights into the role of sleep in more niche aspects of human health. From here, I'd be interested in attempting future projects involving the role of the circadian rhythm on affective disorders such as depression or bipolar disorder.

Considering the state of my project as it is, my goal for the future will be to work to improve the results I have already concluded. I would like to incorporate different methods in order to determine a better-fit model, as well as refine my downsampling process to incorporate more patient data. Also, refining my performance comparisons will be necessary.

## To reproduce results
To reproduce results, ensure your train_series.parquet and train_events.csv have been downsampled to only include every 10th group of unique patient data.
Derive features from the given 'enmo' and 'anglez' in your train_series by finding the maximum, mean, and variance for both values over a rolling period of 10 data points, and store this new data. Merge this with the labeled CSV data at the correct intervals.

Split your new dataframe, which includes your features, into X_test, y_test, X_train, and y_train datasets.

Create a model (I used sequential models for both #1 and #2), and train it on your X_train data. Draft a performance plot for your model.

### Overview of files in repository

  * down+visual.ipnyb : File contains initial downsampling and data visualization methods.
  * FeatureExplore.ipnyb : File contains the methods for obtaining the feature frames of anglez and enmo, including mean, max, and variance.
  * RoughPrediction.ipnyb : File contains multiple models trained on the feature frames, as well as performance plot methods.


### Software Setup
Required packages:

  * Numpy
  * Scikitlearn
  * Keras/Tensorflow
  * Pandas
  * Pyarrow

All packages can be installed through pip
  

### Data

The data necessary to complete this project can be found here:
https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data

## Citations

For initial model:
* https://www.kaggle.com/code/phuwichwinyutrakul/neural-network-child-sleep-pattern/input
  



