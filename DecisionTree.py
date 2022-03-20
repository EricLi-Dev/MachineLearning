# Environment Setup
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
print(melbourne_data.describe())
#               Rooms         Price      Distance      Postcode  ...    YearBuilt     Lattitude    Longtitude  Propertycount
# count  13580.000000  1.358000e+04  13580.000000  13580.000000  ...  8205.000000  13580.000000  13580.000000   13580.000000
# mean       2.937997  1.075684e+06     10.137776   3105.301915  ...  1964.684217    -37.809203    144.995216    7454.417378
# std        0.955748  6.393107e+05      5.868725     90.676964  ...    37.273762      0.079260      0.103916    4378.581772
# min        1.000000  8.500000e+04      0.000000   3000.000000  ...  1196.000000    -38.182550    144.431810     249.000000
# 25%        2.000000  6.500000e+05      6.100000   3044.000000  ...  1940.000000    -37.856822    144.929600    4380.000000
# 50%        3.000000  9.030000e+05      9.200000   3084.000000  ...  1970.000000    -37.802355    145.000100    6555.000000
# 75%        3.000000  1.330000e+06     13.000000   3148.000000  ...  1999.000000    -37.756400    145.058305   10331.000000
# max       10.000000  9.000000e+06     48.100000   3977.000000  ...  2018.000000    -37.408530    145.526350   21650.000000


###          Decision Tree Regression Model      ###
### Predicting Price Given 'Features' of A House ###

#prioritizing columns/variables
print(melbourne_data.columns)
#missing values in house data --> dropping houses from DataFrame
melbourne_data = melbourne_data.dropna(axis=0)

#Column we want to predict
y = melbourne_data.Price
#Columns of house Features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
#              Rooms     Bathroom      Landsize    Lattitude   Longtitude
# count  6196.000000  6196.000000   6196.000000  6196.000000  6196.000000
# mean      2.931407     1.576340    471.006940   -37.807904   144.990201
# std       0.971079     0.711362    897.449881     0.075850     0.099165
# min       1.000000     1.000000      0.000000   -38.164920   144.542370
# 25%       2.000000     1.000000    152.000000   -37.855438   144.926198
# 50%       3.000000     1.000000    373.000000   -37.802250   144.995800
# 75%       4.000000     2.000000    628.000000   -37.758200   145.052700
# max       8.000000     8.000000  37000.000000   -37.457090   145.526350


#The process of building and using a Model:
#Define - What type of model will it be? A decision tree? Some other type of model?
#Fit - Capture patterns from provided data.
#Predict - Predict outcomes.
#Evaluate - Determine the true accuracy of our Model.

#DEFINE : Decision Tree Regression Model
#random_state --> used to ensure same results every time model is used
melbourne_model = DecisionTreeRegressor(random_state=1)

#FIT : Fitting our model with (X) features data and (y) price data
melbourne_model.fit(X, y)

#PREDICT : Exploring our predictions using the fitted model
print("Making predictions for the following 5 houses:")
print(X.head())
print("Predictions Are:")
print(melbourne_model.predict(X.head()))

#EVALUATE : Model Validation
#Common Mistake
# People often make predictions with their TRAINING DATA and compare those predictions
# to the target values in the TRAINING DATA

#Mean Absolute Error (MAE) = abs(actualPrice - predictedPrize)
predicted_home_prices = melbourne_model.predict(X)
print("Mean Absolute Error:")
print("Not Properly Split: " + str(mean_absolute_error(y, predicted_home_prices)))

#Splitting our original melbourne house data to have [validation data]
#data A can be used for fitting the Model
#data B can then be used to test for accuracy

#Splits X and y data for 'training' or 'validating'
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Define model
melbourne_model = DecisionTreeRegressor(random_state = 1)
# Fit model
melbourne_model.fit(train_X, train_y)
# Evaluate Data
val_predictions = melbourne_model.predict(val_X)
print("Properly Split: " + str(mean_absolute_error(val_y, val_predictions)))
