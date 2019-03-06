'' LASSO Regression, also known as Least Absolute Shrinkage and Selection Operator is a regression method which can be
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