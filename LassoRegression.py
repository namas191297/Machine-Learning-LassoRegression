''' LASSO Regression, also known as Least Absolute Shrinkage and Selection Operator is a regression method which can be
defined as a method that is used to determine the values of the variables in such a way that the
value of :

Lambda * | Sum of the weights/slope |   (L1 Norm)

is minimized.

This means that the penalty is introduced in order to reduce the variance of the model. Additionally, what LASSO
regression does is that curve or the line that approximates the mapping function can shrink the value of slope/weight = 0,
unlike ridge regression, in which the slope of the line can only be compressed due to the least squares method, and is always
asymptotically close to 0, but never 0.
This is where Sparsity is introduced in the model, which means that the parameters or the variables that are useless in
the function are shrunk to the value 0 and hence these variables are eliminated from the model completely, that prevents
overfitting of the model. '''

''' Problem : Consider that we need to predict the size of an animal based on the following equation give:
  
  Size = CONSTANT + slope * WIEGHT + intake * DIETTYPE + astrologicaloffset * Sign + airspeescalar * AirspeedOfSwallow
  
  CONSTANT / Y-INTERCEPT = 18
  
  As we all know that in terms of calculating the size of an animal, WEIGHT and DIETTYPE are helpful variables and
  that might help us in the prediction of the size. However, the zodiac sign and the airspeed of a swallow are 
  unrelated and useless variables in this equation. As we know the LASSO regression is used to bring Sparsity and
  eliminate the useless variables, hence we use the LASSO Regression.
  
  Consider the equation to be : 
  
  SIZE = 18 + 7*WEIGHT + 20*DIETTYPE -0.5*SIGN + 12.5*AirspeedOfSwallow
   
  '''

#Importing our dependencies
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import *
import random

#Defining certain variables

TOTAL_SAMPLE_COUNT = 100
CONSTANT = 18
TRAIN_SAMPLE_COUNT = int(TOTAL_SAMPLE_COUNT*(30/100))
TEST_SAMPLE_COUNT = TOTAL_SAMPLE_COUNT - TRAIN_SAMPLE_COUNT
MAX_SAMPLE_SIZE = 70
INVOKE_ERROR = 0.04

#Defining functions to generate values
def weight(x):
    res = x * 7
    error = random.uniform(-INVOKE_ERROR,INVOKE_ERROR)
    return res + (res*error)

def diet(x):
    res = x * 20
    error = random.uniform(-INVOKE_ERROR, INVOKE_ERROR)
    return res + (res * error)

def sign(x):
    res = x * (-0.5)
    error = random.uniform(-INVOKE_ERROR, INVOKE_ERROR)
    return res + (res * error)

def airSpeed(x):
    res = x * 12.5
    error = random.uniform(-INVOKE_ERROR, INVOKE_ERROR)
    return res + (res * error)

#Generating the values

number_list = np.random.randint(TOTAL_SAMPLE_COUNT,size=MAX_SAMPLE_SIZE).astype(float)
weights = weight(number_list)
diets = diet(number_list)
signs = sign(number_list)
airSpeeds = airSpeed(number_list)
size = weights + diets + signs + airSpeeds + CONSTANT

#Dividing data into training and testing

weights_train, weights_test = np.split(weights,[TRAIN_SAMPLE_COUNT,])
diets_train, diets_test = np.split(diets,[TRAIN_SAMPLE_COUNT,])
signs_train, signs_test = np.split(signs,[TRAIN_SAMPLE_COUNT,])
airSpeeds_train, airSpeeds_test = np.split(airSpeeds,[TRAIN_SAMPLE_COUNT,])
sizes_train, sizes_test = np.split(size, [TRAIN_SAMPLE_COUNT,])

#Creating dictionaries to be converted to Pandas Dataframe

independent_train_dict = {"WEIGHTS" : weights_train,
                     "DIETS" : diets_train,
                     "SIGNS" : signs_train,
                     "AIRSPEEDS" : airSpeeds_train}

independent_test_dict = {"WEIGHTS" : weights_test,
                     "DIETS" : diets_test,
                     "SIGNS" : signs_test,
                     "AIRSPEEDS" : airSpeeds_test}

dependent_train_dict = {"SIZES" : sizes_train}

dependent_test_dict = {"SIZES" : sizes_test}


#Converting the created dictionaries into Pandas dataframe use them in the model

independent_train_data = pd.DataFrame(data=independent_train_dict)
independent_test_data = pd.DataFrame(data=independent_test_dict)
dependent_train_data = pd.DataFrame(data=dependent_train_dict)
dependent_test_data = pd.DataFrame(data=dependent_test_dict)

#Initializing the model and training our datas

model = linear_model.Lasso(alpha=0.01)
model.fit(independent_train_data,dependent_train_data)

#Printing scores and the prediction
predictions = model.predict(independent_test_data)
print("Predicting random data for values (5,7.5,-2,6) ........")
print(model.predict([[5,7.5,-2,6]]))
print("Model Score:",model.score(independent_test_data,dependent_test_data))
print("Coefficients:",model.coef_)
print("Y-intercept:",model.intercept_)







