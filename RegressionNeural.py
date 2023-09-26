# Dataset Description
# holiday:Categorical US National holidays plus regional holiday, Minnesota State Fair
# temp Numeric Average temp in kelvin
# rain_1h Numeric Amount in mm of rain that occurred in the hour
# snow_1h Numeric Amount in mm of snow that occurred in the hour
# clouds_all Numeric Percentage of cloud cover
# weather_main Categorical Short textual description of the current weather
# weather_description Categorical Longer textual description of the current weather
# date_time DateTime Hour of the data collected in local CST time
# traffic_volume Numeric Hourly I-94 ATR 301 reported westbound traffic volume

# Import required packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib
#import matplotlib.pyplot as plt
# Set up the plot environment
matplotlib.use('Qt5Agg')

# Filter warnings in our console (NOT ADVISABLE, a bug in the pandas latest version creates excessive warnings)
import warnings
warnings.filterwarnings("ignore")

# Read the titanic csv
df = pd.read_csv('Data Analytics//Week5/bank.csv')

# Print the dataset
df

# Print the 1st row
df.iloc[0]

# TODO QUIZ 1
# Which features will we include in our classifier?
# holiday, weather date/time, weather description...
# TODO QUIZ 2
# Which ones do we need to encode in numerical values?
# holiday, weather main basically the categorical ones
# Convert string of holiday column to numbers with pandas
#df['holiday'] = pd.factorize(df['holiday'])[0]

# Convert string of weather_main column to numbers with pandas
#df['weather_main'] = pd.factorize(df['weather_main'])[0]

# Convert string of weather_description column to numbers with pandas
#df['weather_description'] = pd.factorize(df['weather_description'])[0]

# TODO QUIZ 3
# What about time as a feature?
# yes the weekend and monday mornings will count.
# TODO QUIZ 4
# How do we format it?
# shown below
# Examine the type
#type(df['date_time'][0])

# Since it is a string we need to convert it to date-time format
#df['date_time'] = pd.to_datetime(df['date_time'])

# We only want the hour of each day
#df['hour'] = df['date_time'].dt.hour

# Convert string of Sex column to numbers with pandas
#df['Embarked'] = pd.factorize(df['Embarked'])[0]
# Factorize the 'string' column and store the dictionary in a variable
dict = pd.factorize(df['job'])
df['job'] = pd.factorize(df['job'])[0]
# Factorize the 'string' column and store the dictionary in a variable
dict2 = pd.factorize(df['marital'])
df['marital'] = pd.factorize(df['marital'])[0]
dict3 = pd.factorize(df['age'])
df['age'] = pd.factorize(df['age'])[0]
dict4 = pd.factorize(df['education'])
df['education'] = pd.factorize(df['education'])[0]
dict5 = pd.factorize(df['default'])
df['default'] = pd.factorize(df['default'])[0]
dict6 = pd.factorize(df['balance'])
df['balance'] = pd.factorize(df['balance'])[0]
dict7 = pd.factorize(df['housing'])
df['housing'] = pd.factorize(df['housing'])[0]
dict8 = pd.factorize(df['loan'])
df['loan'] = pd.factorize(df['loan'])[0]
dict9 = pd.factorize(df['contact'])
df['contact'] = pd.factorize(df['contact'])[0]
dict10 = pd.factorize(df['termdeposit'])
df['termdeposit'] = pd.factorize(df['termdeposit'])[0]

df = df.astype(float)

# Select only the columns of features
X = df[['age',
        'job',
        'marital',
        'education',
        'default',
        'balance',
        'housing',
        'loan',
        'contact']]

# Select the column of prediction class
Y = df['termdeposit']

# Ten-fold Cross Validation Split
# We want 10 splits, at a set starting point
kfold = KFold(n_splits=10, shuffle=True, random_state=804)

# Enumarator to be used inside the loop for reporting
FoldNum = 1

# Create a list to store Squared Error scores
SqEr = []

# Loop through each fold
for train, test in kfold.split(X, Y):
    # Print the number of fold we are currently fitting
    print('Fold:',FoldNum)
    # Fit the Linear Classifier to our train subset
    clf = LinearRegression().fit(X.iloc[train], Y.iloc[train])
    # Predict the classes for our test subset
    Y_pred = clf.predict(X.iloc[test])
    # Add the accuracy score as the next element of our accuracy scores list
    SqEr.append(
        mean_squared_error(Y.iloc[test], Y_pred,squared = False)
    )
    # Increase FoldNum value by one
    FoldNum += 1

# TODO Now let's do the same prediction task with a neural network

# Import the appropriate packages
from keras.models import Model
from keras.layers import Input,Dense
from tensorflow.keras import backend

# We need to format the X into a simple row of vector values
X = X.values

# Enumarator to be used inside the loop for reporting
FoldNum = 1

# Loop through each fold
for train, test in kfold.split(X, Y):
    # Define the input, with the number of features
    input = Input(shape=(8,))
    # Define the Neural Single layer, and what will be fed into the layer
    dense = Dense(4, activation='relu')(input)
    # Define the number of final activation neurons, and what will be fed into the layer
    output = Dense(1)(dense)
    # Define the Model, with our predefined inputs and outputs
    model = Model(inputs=input, outputs=output)
    # Define the compile parameters of our model
    # Adam is the best performing stochastic gradient descend optimizer
    # Mean squared error is used as the loss function
    # The root of the mean squared error is used as metric to evaluate the progress of our training
    model.compile(optimizer='adam', loss='mean_squared_error', metrics = ['RootMeanSquaredError'])
    # Fit the model and save the respective training and validation values per epoch in the history variable
    # 10 number of epochs is a good starting point
    # I have set the batch size as 64 for faster training, lower batch might produce slightly better results
    # Verbose can be set as 'auto', 0, 1, or 2. Different levels of information per epoch displayed
    # Validation split defines the percentage of training dataset that we will use as validation
    # We use 11% of the 90% of the training subset, 9.9% of the initial dataset size
    history = model.fit(X[train], Y[train], epochs=10, batch_size=64, verbose=1, validation_split=0.11)
    # Evaluate our model in the test subset
    scores = model.evaluate(X[test], Y[test], verbose=1)
    # Print the number of fold we are currently fitting
    print('Fold:',FoldNum, 'is complete.')
    # Increase FoldNum value by one
    FoldNum += 1

# Clear the trained models from computer memory
backend.clear_session()

# Plot the training root mean squared error metric values per epoch
plt.plot(history.history['root_mean_squared_error'])
# Plot the validation root mean squared error metric values per epoch
plt.plot(history.history['val_root_mean_squared_error'])
# Add a title to the plot
plt.title('Model MSE')
# Add a label to Y axis
plt.ylabel('Root Mean Squared Error')
# Add a label to X axis
plt.xlabel('Epoch')
# Add a legend to the plot
plt.legend(['training', 'validation'])
# Show the interactive plot
plt.show() # graph shows that the more we train the data the more closer the validation of the data is.

# Print the mean root squared error of our linear regression
print('The mean root square error of our linear regression is:', np.mean(SqEr))

# Print the mean root squared error of our linear regression
print('The mean root square error of our neural network is:', scores[1])