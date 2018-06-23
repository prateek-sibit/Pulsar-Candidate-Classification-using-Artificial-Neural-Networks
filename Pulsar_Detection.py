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

# Now the sigmoid function returns probabilties 
# We must map these probabilities into binary (0/1) values
def prob_to_binary(value):
    if(value>=0.5):
        return 1
    else:
        return 0

vector_f = np.vectorize(prob_to_binary)
y_predict = vector_f(y_predict)

# Results and Metrics
from sklearn.metrics import confusion_matrix

# Creating the confusion matrix for the model
cm = confusion_matrix(y_test,y_predict)
# Calculating the accuracy of the model using the confusion matrix = (TP+TN)/(TP+FP+TN+FN)
accuracy = np.sum(np.diag(cm)/np.sum(cm))

# Saving the Results and model
classifier.save('Initial_ANN.h5')
file = open('Initial_results.txt','w')
results = 'Confusion Matrix : \n'+str(cm)+'\nAccuracy : '+str(accuracy)
file.write(results)
file.close()

# Evaluating, Improving and Tuning the ANN

# Evaluating the Average accuracy of the current model using k-folds cross validation
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

def build_classifier():
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
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
average_accuracy = np.mean(accuracies)

# Saving to the results file
file = open('Initial_results.txt','a')
average = '\nAccuracies for Cross validation : \n'+str(accuracies)+'\n\nAverage Accuracy : '+str(average_accuracy)
file.write(average)
file.close()

# Improving the ANN using Grid Search and Dropout regularization
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# Building the classifier function with added improvements
def build_classifier(optimizer):
    # Initializing the ANN
    classifier = Sequential()
    # Building the input layer
    classifier.add(Dense(input_dim=8,units=5,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    # Adding a second layer
    classifier.add(Dense(units=5,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    # Adding a third layer
    classifier.add(Dense(units=5,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    # Adding a fourth layer
    classifier.add(Dense(units=5,activation='relu',kernel_initializer='uniform'))
    classifier.add(Dropout(0.1))
    # Creating the output layer
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
    # Compiling the ANN
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn=build_classifier)
# Create a dictionary of hyperparameters to tune
parameters = {'batch_size' : [10,25],
              'epochs' : [100,500],
              'optimizer':['adam','rmsprop']}

# Creating the grid search object
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_score = grid_search.best_score_