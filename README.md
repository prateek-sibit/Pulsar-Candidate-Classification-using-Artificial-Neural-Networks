# Pulsar-Candidate-Classification-using-Artificial-Neural-Networks

## About the Files in this Project

- HTRU_2.csv : The Dataset file in csv (comma separated values) format

- Initial_ANN.h5 : It is the initial model with the configuration and weights.
  This file can be loaded directly to use as a model without going through the training using the commands
```
from keras.models import load_model
classifier = load_model('Initial_ANN.h5')
```

- Initial_results.txt : Text file that contains the initial results and the accuracies evaluated using k-folds cross validation

- Pulsar_Detection.py : Main python file containing the project code

 
## Dataset Information :

HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey. 

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the inter-stellar medium, and states of matte. 

As pulsars rotate, their emission beam sweeps across the sky, and when this crosses our line of sight, produces a detectable pattern of broadband radio emission. As pulsars 
rotate rapidly, this pattern repeats periodically. Thus pulsar search involves looking for periodic radio signals with large radio telescopes. 

Each pulsar produces a slightly different emission pattern, which varies slightly with each rotation. Thus a potential signal detection known as a 'candidate', is averaged over many rotations of the pulsar, as determined by the length of an observation. In the absence of additional info, each candidate could potentially describe a real pulsar. However in practice almost all detections are caused by radio frequency interference (RFI) and noise, making legitimate signals hard to find. 

Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted, 
which treat the candidate data sets as binary classification problems. Here the legitimate pulsar examples are a minority positive class, and spurious examples the majority negative class. At present multi-class labels are unavailable, given the costs associated with data annotation. 

The data set shared here contains 16,259 spurious examples caused by RFI/noise, and 1,639 real pulsar examples. These examples have all been checked by human annotators. 

Candidates are stored in both files in separate rows. Each row lists the variables first, and the class label is the final entry. The class labels used are 0 (negative) and 1 (positive). 

## Attribute Information 

Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency (see [3] for more details). The remaining four variables are similarly obtained from the DM-SNR curve (again see [3] for more details). These are summarised below: 

1. Mean of the integrated profile. 
2. Standard deviation of the integrated profile. 
3. Excess kurtosis of the integrated profile. 
4. Skewness of the integrated profile. 
5. Mean of the DM-SNR curve. 
6. Standard deviation of the DM-SNR curve. 
7. Excess kurtosis of the DM-SNR curve. 
8. Skewness of the DM-SNR curve. 
9. Class 

## Results 
The Initial Model yields the following results
```
Confusion Matrix : 
[[4883    1]
 [ 413   73]]
Accuracy : 0.9229050279329608

Average Accuracy after Cross Validation : 0.9727022514088827
```


## Citations :
*R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656*

*R. J. Lyon, HTRU2, DOI: 10.6084/m9.figshare.3080389.v1*
