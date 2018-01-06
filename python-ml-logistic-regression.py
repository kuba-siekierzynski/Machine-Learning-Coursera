"""
Machine Learning with Python
Logistic regression

Coded by Kuba Siekierzynski (c) 2018

The code is the second of the series inspired by the awesome Andrew Ng's video course on machine learning:
http://www.coursera.org/learn/machine-learning

This code shows an implementation of logistic regression, the machine learning algorithm to predict discrete values.
"""

# suppress the warnings when logarithm calculates infinity
import warnings
warnings.simplefilter("ignore", RuntimeWarning)

# The problem to be solved:
# We have forty samples of fish, that we have measured on two dimensions - weight and length. Based on the measurements,
# half of them was categorized as 'good' and half as 'bad' (represented by 'o' and 'x', respectively). We now want to
# work out a model able to predict, which category will a newly caught fish be, given its measurements.

import numpy as np
print('Welcome to Machine Learning with Python!')
print('Lesson 2: Logistic regression')
print('\n'+40*'=')

print('The task now is to create a model which would separate good fish from the bad. Too small or weirdly built fish'
      'are harder to process and so we want to see which are the ones we may make profit from.\n')

# data representing the measurements and category 1 and 2 (or 'o' and 'x')
data =\
[(10, 12, 1), (15, 14, 1), (6, 18, 1), (12, 10, 2),\
(17, 13, 1), (18, 18, 1), (11, 15, 1), (3, 17, 1),\
(5, 15, 1), (16, 9, 1), (16, 16, 1), (24, 16, 1),\
(22, 18, 1), (25, 19, 1), (27, 17, 1), (26, 10, 1),\
(18, 12, 1), (24, 7, 1), (23, 12, 1), (28, 9, 2),\
(7, 2, 2), (6, 8, 2), (2, 3, 2), (14, 1, 2),\
(7, 5, 2), (20, 1, 2), (2, 9, 1), (10, 4, 2),\
(2, 7, 2), (4, 1, 2), (13, 2, 2), (10, 1, 2),\
(4, 5, 2), (17, 3, 2), (9, 6, 2), (12, 5, 1),\
(5, 3, 2), (14, 4, 2), (23, 2, 2), (17, 1, 2)]

# Just like in the case of the linear regression model, we want build a model able to predict the fish category based on
# the measurement values. We thus have to convert the data to the matrix type.

# X matrix will hold the measurements - remember to add a column of '1's for the intercept term
dataX = np.matrix(data)[:,:2]
X = np.ones((len(dataX),3))
X[:,1:] = dataX

# y matrix will hold the labels - but we have to convert them into 0 and 1 (bad or good)
y = 2-np.matrix(data)[:,2]

'''
Below is the show() function, which will use the ASCII to graphically "plot" the charts
'''

def show(arr, p):
    for j in range(np.size(arr,1)-1, -1, -1):
        print('{:>2}'.format(j if j%5 == 0 else ' ')+' '+chr(179+(j%5==0)), end='')
        for i in range(np.size(arr,0)):
            print(p[arr[i][j]], end='')
        print()
    print('   '+chr(192)+(chr(194)+chr(196)*4)*(np.size(arr,0)//5))
    print('    ', end='')
    print(*list(('{:<4}'.format(i*5) for i in range(np.size(arr,0)//5))))
    print()

# t and p are the plot variables
t = np.zeros((30,20), np.int8)
p = [' ', '#', 'o', 'x']

# Populating the plot array from the measurement data
for i in data:
    t[i[0]][i[1]] = i[2]+1

# show is displaying a chart with the fish data points
print('Plotting the datapoints:', 'o - good fish', 'x - bad fish', sep='\n')
show(t, p)

'''
In contrast to linear regression from Lesson 1, we are now not looking for a model predicting the continous values, but
rather a classificator able to predict categories (or labels). In this case it is binary classification (fish is either
of two states only). So, we will be searching for a _probability_ that a given sample is of either of the two states.

We have to use logistic (or sigmoid) function and plug our regular X*theta function into it. So it will be of a format:

g(z) = 1 / (1 + e^(-z)), which in the numpy matrix format will be as below:

'''

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
In this relatively simple situation, we will assume that whenever the hypothesis is greater than 0.5, the prediction
will return 1 and if it is lower than 0.5 - it will return 0. The situation where the hypothesis is equal to 0.5 is a
so-called "decision boundary" - which is exactly the thing we are looking for. Our model is going to find that boundary
and according to it, it will make predictions on future, unseen fish.

'''

print('It can be observed that there is no possibility to draw a straight line which would separate the two groups.'
      'We are thus looking for a decision boundary like this (a sample)')

# a sample decision boundary
for i in range(np.size(t, 0)):
    t[i][13-int(i/2.5)] = 1

show(t, p)

'''
The decision boundary is where the cost function is the lowest possible, so again we are to optimize it. However, this
time it will not be calculated as a square of errors anymore. Since our primary function is a logistic function, its
cost function will be of a logarithmic shape. The concrete formula is as below:

pred = sigmoid(X*theta)
J = 1/m * sum(-y*log(pred) - (1-y)*log(1-pred))

or in numpy, as below:
'''

def J(X, y, theta):
    theta = np.matrix(theta).T # we again need a transposed matrix theta
    m = len(y) # m is the number of datapoints
    predictions = sigmoid(X * theta) # stores the sigmoid predictions, translated into binary classes
    l0 = np.log(predictions.T)
    l1 = np.log(1-predictions.T)
    l0[l0==-np.inf] = 0
    l0[l0==np.inf] = 0
    l1[l1==-np.inf] = 0
    l1[l1==np.inf] = 0
    return -1/(m * np.sum(y*l0 + (1-y)*l1)), 1/m * (X.T * (predictions-y)) # the second parameter is the gradient

print('\nLet\'s find that theta vector!')

# let's check the cost if we would assume theta at two different values
print('\nChecking two example cases of theta:')
for th in [0, 0, 0], [1, 2, 3]:
    print('Assuming theta vector at {}, the cost would be {:.12f}'.format(th, J(X, y, th)[0].item()))

'''
The optimal theta vector will be needed for our model to predict with the smallest possible error.
Assuming that J is a cost function, this is again an optimization problem - we need to find the minimum of J.
The gradient descent function looks EXACTLY the same as for linear regression! This time, however, our hypothesis is
a sigmoid function.

'''

# gradient descent function will iteratively update theta by a small fraction alpha (also called the learning rate) for
# a number of iterations

def gradient(X, y, alpha, theta, iters):
    J_history = np.zeros(iters) # will store historical values of J for each iteration
    m = len(y) # m is the number of datapoints
    theta = np.matrix(theta).T # theta has to be transposed again
    for i in range(iters):
        h0 = sigmoid(X * theta) # zero hypothesis for each datapoint
        delta = (1 / m) * (X.T * h0 - X.T * y) # the gradient descent
        theta = theta - alpha * delta # update theta by learning rate times gradient
        J_history[i] = J(X, y, theta.T)[0] # save the J of a particular iteration, it should drop in the next
    return J_history, theta # return the history of J plus the optimal theta

print('\n'+40*'=')

theta = np.matrix([np.random.random(),np.random.random(),np.random.random()]) # we initialize theta at random values
alpha = 0.005 # learning rate - if too low, the algorithm will not converge, if too high, it can "explode"
iters = 20000 # number of iterations - reduce if "Time limit exceeded"
print('\n== Model summary ==\nLearning rate: {}\nIterations: {}\nInitial theta:\n{}\nInitial J: {:.12f}\n'.\
      format(alpha, iters, theta, J(X,y,theta)[0].item()))

print('Training the model...', end='')
J_history, theta_min = gradient(X, y, alpha, theta, iters)
print('Done.')

print('\nFinal theta:\n{}\nFinal J: {:.12f}'.format(theta_min.T, J(X,y,theta_min.T)[0].item()))

'''
Now that we have the model trained, let's check if it works. We will plot a chart with the decision boundary calculated
by the model and on top of that, we will plot real datapoints (our training set). To do that, we will iteratively cast
a 'predict' function (defined below) on every possible measurement pair.
'''

def predict(a,b,theta_):
    return np.round(sigmoid(np.matrix([1,a,b])*theta_))

print('\n'+40*'=')

# db will store the chart for displaying the decision boundary
db = np.zeros((30,20), dtype=np.int8)
for j in range(np.size(db, 1)):
    for i in range(np.size(db, 0)):
        db[i][j] = predict(i,j,theta_min)
        # np.round(sigmoid(np.matrix([1,i,j])*theta_min))

# Populating the plot array from the measurement data
for i in data:
    db[i[0]][i[1]] = i[2]+1

# let's show it now
print('\nEverything in the dotted area will be predicted by the model as a "good" fish, while datapoints belonging to'
      'the blank area will be predicted as "bad" ones.')
show(db, [' ', '.', 'o', 'x'])


'''
Some of the good fish were predicted as bad ones and vice-versa. It would still be wise to know what accuracy our model
has (remember - anything better than a pure-luck 50% is great :)
'''

def score(X, y, theta):
    return np.mean(np.round(sigmoid(np.matrix(X)*theta))==y), np.sum(np.round(sigmoid(np.matrix(X)*theta))==y)

print('Model accuracy {:.1%} ({}/{} points)'.format(score(X, y, theta_min)[0], score(X, y, theta_min)[1], len(y)))

print('Feel free to modify the datapoints and see what happens! :)')

print('\n'+40*'=')
print('\nNOTE: The code initializes the model with different theta each time, the results may very between runtimes.')