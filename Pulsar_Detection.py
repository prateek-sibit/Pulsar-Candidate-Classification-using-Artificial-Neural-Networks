# Pulsar Candidate Classification using ANN

# Importing Data Handling libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams

# Setting Visualization settings
sb.set_style('darkgrid')
rcParams['figure.figsize'] = (14,7)
rcParams['font.size'] = 12

# Reading the Data from csv 

# Note that the data contains no positional information or other astronomical details. 
# It is simply feature data extracted from candidate files using the PulsarFeatureLab tool
# So we specify header=None in the Data reading
 
data = pd.read_csv('HTRU_2.csv',header=None)

# We do not do any Exploratory Data analysis as it is not straightforward to interpret 
# visualising the attributes of the dataset

# Data Preprocessing
# Defining the Dependent and Independent variables
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Making the Artificial Neural Network

# Import keras and other packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Building the input layer
classifier.add(Dense(input_dim=8,units=5,activation='relu',kernel_initializer='uniform'))

# Adding a second layer
classifier.add(Dense(units=5,activation='relu',kernel_initializer='uniform'))

# Adding a third layer
classifier.add(Dense(units=5,activation='relu',kernel_initializer='uniform'))

# Creating the output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the model on the training set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)

# Making the predictions 
y_predict = classifier.predict(X_test)

# Results and Metrics
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_predict)
accuracy = np.sum(np.diag(cm)/np.sum(cm))