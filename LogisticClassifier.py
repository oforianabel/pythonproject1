# Import required packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Delete the following 3 line if you are on MacOS
# Set up the plot environment
#import matplotlib
#matplotlib.use('Qt5Agg')

# Filter warnings in our console (NOT ADVISABLE, a bug in the pandas latest version creates excessive warnings)
import warnings
warnings.filterwarnings("ignore")

# Read the titanic csv
df = pd.read_csv('Data Analytics//Week 4//bank.csv')

# Print the dataset
df

# Print the 1st row
df.iloc[0]

# survival 	Survival 	0 = No; 1 = Yes
# pclass 	Passenger Class 	1 = 1st; 2 = 2nd; 3 = 3rd
# name 	First and Last Name
# sex 	Sex
# age 	Age
# sibsp 	Number of Siblings/Spouses Aboard
# parch 	Number of Parents/Children Aboard
# ticket 	Ticket Number
# fare 	Passenger Fare
# cabin 	Cabin
# embarked 	Port of Embarkation 	C = Cherbourg; Q = Queenstown; S = Southampton

# TODO QUIZ 1
# Which will be our prediction class?

# TODO QUIZ 2
# Which features will we include in our classifier?

# TODO QUIZ 3
# Which features will we need to encode into numbers, aka factorize?

# Convert string of Sex column to numbers with pandas
#df['Sex'] = pd.factorize(df['Sex'])[0]

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

f = dict2[1][2]
print(f)

# TODO QUIZ 4
# Is Cabin column ready to be used for our analysis?

# Let's have a look
df['job'].unique()

# We can replace NaN values with any other value
# df = df.fillna('ZZZ')
# Cabin column has a bunch of alphanumeric characters

# TODO QUIZ 5
# Can we think of any ways we can make use of these values?


# TODO ONLY IN EXTREME CASES
# We can remove a whole column
#del df['Cabin']
# We can remove rows with NaN values
#df = df.dropna()


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

# This time we will perform a proper ten-fold Cross Validation
# We want 10 splits, at a set starting point
kfold = KFold(n_splits=10, shuffle=True, random_state=804)

# Initialize a Logistic Classifier
clf = LogisticRegression(random_state=804)

# Enumarator to be used inside the loop for reporting
FoldNum = 1

# Create a list to store accuracy scores
AccScores = []

# Create a list to store log-likehood errors
LogLikeErr = []

# Loop through each fold
for train, test in kfold.split(X, Y):
    # Print the number of fold we are currently fitting
    print('Fold:',FoldNum)
    # Fit the Logistic Classifier to our train subset
    clf.fit(X.iloc[train], Y.iloc[train])
    # Predict the classes for our test subset
    Y_pred = clf.predict(X.iloc[test])
    # Add the accuracy score as the next element of our accuracy scores list
    AccScores.append(accuracy_score(Y.iloc[test], Y_pred))
    # Add the log error as the next element of our log-likehood errors list
    LogLikeErr.append(log_loss(Y.iloc[test], Y_pred))
    # Increase FoldNum value by one
    FoldNum += 1

# We can also plot the accuracy and error values
# Plot a scatter plot of (x,y) pairs of values
plt.scatter(LogLikeErr,AccScores)
# Set the x-axis label
plt.xlabel('Log-Likehood error')
# Set the y-axis label
plt.ylabel('Accuracy')
# Show the plot
plt.show()
# as log error increases, accuracy decreases.
# highest accuracy achieved was 0.855