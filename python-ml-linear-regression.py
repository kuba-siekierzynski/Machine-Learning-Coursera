"""
Machine Learning with Python
Linear regression

Coded by Kuba Siekierzynski (c) 2017

The code is the first of the series inspired by the awesome Andrew Ng's video course on machine learning:
http://www.coursera.org/learn/machine-learning
This code shows an implementation of linear regression, the initial machine learning algorithm to predict continuous values.
"""

# The problem to be solved:
# We have trucks located in different cities and each truck brings a profit or loss. We have the historical data and
# determined that the profit depends on the city's population. We want to find this relation.

import numpy as np
print('Welcome to Machine Learning with Python!')
print('Lesson 1: Linear regression')
print('\n'+40*'=')

# data contains the city population (in 10,000s) in the first column
# and the profit/loss (in 10,000$) in the second columns
# the data was rescaled to save on calculations and resources consumption

# Based on the first entry, a truck in a city of population of 61,101 brought a profit of $175,920
data =\
[
[6.1101,17.592],
[5.5277,9.1302],
[8.5186,13.662],
[7.0032,11.854],
[5.8598,6.8233],
[8.3829,11.886],
[7.4764,4.3483],
[8.5781,12],
[6.4862,6.5987],
[5.0546,3.8166],
[5.7107,3.2522],
[14.164,15.505],
[5.734,3.1551],
[8.4084,7.2258],
[5.6407,0.71618],
[5.3794,3.5129],
[6.3654,5.3048],
[5.1301,0.56077],
[6.4296,3.6518],
[7.0708,5.3893],
[6.1891,3.1386],
[20.27,21.767],
[5.4901,4.263],
[6.3261,5.1875],
[5.5649,3.0825],
[18.945,22.638],
[12.828,13.501],
[10.957,7.0467],
[13.176,14.692],
[22.203,24.147],
[5.2524,-1.22],
[6.5894,5.9966],
[9.2482,12.134],
[5.8918,1.8495],
[8.2111,6.5426],
[7.9334,4.5623],
[8.0959,4.1164],
[5.6063,3.3928],
[12.836,10.117],
[6.3534,5.4974],
[5.4069,0.55657],
[6.8825,3.9115],
[11.708,5.3854],
[5.7737,2.4406],
[7.8247,6.7318],
[7.0931,1.0463],
[5.0702,5.1337],
[5.8014,1.844],
[11.7,8.0043],
[5.5416,1.0179],
[7.5402,6.7504],
[5.3077,1.8396],
[7.4239,4.2885],
[7.6031,4.9981],
[6.3328,1.4233],
[6.3589,-1.4211],
[6.2742,2.4756],
[5.6397,4.6042],
[9.3102,3.9624],
[9.4536,5.4141],
[8.8254,5.1694],
[5.1793,-0.74279],
[21.279,17.929],
[14.908,12.054],
[18.959,17.054],
[7.2182,4.8852],
[8.2951,5.7442],
[10.236,7.7754],
[5.4994,1.0173],
[20.341,20.992],
[10.136,6.6799],
[7.3345,4.0259],
[6.0062,1.2784],
[7.2259,3.3411],
[5.0269,-2.6807],
[6.5479,0.29678],
[7.5386,3.8845],
[5.0365,5.7014],
[10.274,6.7526],
[5.1077,2.0576],
[5.7292,0.47953],
[5.1884,0.20421],
[6.3557,0.67861],
[9.7687,7.5435],
[6.5159,5.3436],
[8.5172,4.2415],
[9.1802,6.7981],
[6.002,0.92695],
[5.5204,0.152],
[5.0594,2.8214],
[5.7077,1.8451],
[7.6366,4.2959],
[5.8707,7.2029],
[5.3054,1.9869],
[8.2934,0.14454],
[13.394,9.0551],
[5.4369,0.61705]
]

# We want to make a model able to predict the profit/loss, based on a given population. In order to do some machine
# learning, the data has to be of a matrix type.

# X matrix will hold city population
X = np.matrix(data)[:,0]

# y matrix will hold the profit/loss information
y = np.matrix(data)[:,1]

'''
Basically, we are looking for a function f(x) returning the _output_ value y based on its _input_ x. We assume a linear
y = ax + b dependence, but it as well might have been a polynominal or any other function. So, we are looking for such
a and b values that give us a function that will somehow reflect the profit based on the population. Like this:

    predicted_profit = a * city_population + b

A quick look at the data shows that it is impossible to find a line which would cross all the datapoints. So, we want to
have the best possible fit. How do we measure the quality of it? The best possible fit is such that makes the smallest
prediction error on the whole dataset. The single error is calculated as the square of the difference between the real
and predicted value, so the total error will simply by the sum of all single ones.

We thus need a so-called cost function which would return the average error of a given f(x) when trying to explain the
datapoints and make predictions. In order to make things quicker, we will look for a vector 'theta', containing the
'a' and 'b' (or more, for more complicated models - theta0, theta1, theta2,...) parameters.
'''

print('\nLooking for y=a*x+b function (a,b=theta)')

# function J calculates the cost under a given set of theta parameters
def J(X, y, theta):
    theta = np.matrix(theta).T # we need a transposed matrix theta
    m = len(y) # m is the number of datapoints
    predictions = X * theta # stores the outputs predicted by f(x) with a given theta as parameter vector
    sqError = np.power((predictions-y),[2]) # a matrix of squared errors between predictions and real values
    return 1/(2*m) * sum(sqError) # the value of the cost function J


# the transformation below adds a column of ones to the left of the X matrix, for calculation reasons
dataX = np.matrix(data)[:,0:1]
X = np.ones((len(dataX),2))
X[:,1:] = dataX

# let's check the cost if we would assume theta at two different values
print('\nChecking two example cases of theta:')
for t in [0,0], [-1,2]:
    print('Assuming theta vector at {}, the cost would be {:.2f}'.format(t, J(X, y, t).item()))  # 32.073, 54.242

'''
Now, how to find the optimal theta vector for our model to predict with the smallest possible error?
Assuming that J is a cost function, this is an optimization problem - we need to find the minimum of J.
We will use a technique called gradient descent - we will initialize theta at all-zeros and gradually move along the J
curve updating all thetas (simultaneously) by small fractions. If J increases - we are going the wrong way, if it
decreases - we are moving along this way.
'''

# gradient descent function will iteratively update theta by a small fraction alpha (also called the learning rate)
# for a number of iterations

def gradient(X, y, alpha, theta, iters):
    J_history = np.zeros(iters) # will store historical values of J for each iteration
    m = len(y) # m is the number of datapoints
    theta = np.matrix(theta).T # theta has to be transposed again
    for i in range(iters):
        h0 = X * theta # zero hypothesis for each datapoint
        delta = (1 / m) * (X.T * h0 - X.T * y) # the gradient descent
        theta = theta - alpha * delta # update theta by learning rate times gradient
        J_history[i] = J(X, y, theta.T) # save the J of a particular iteration, it should drop in the next
    return J_history, theta # return the history of J plus the optimal theta

print('\n'+40*'=')

# we have the function ready, let's do some machine learning!
theta = np.matrix([np.random.random(),np.random.random()]) # we initialize theta at random values
alpha = 0.01 # learning rate - if too low, the algorithm will not converge, if too high, it can "explode"
iters = 2000 # number of iterations - reduce if "Time limit exceeded"



print('\n== Model summary ==\nLearning rate: {}\nIterations: {}\nInitial theta: {}\nInitial J: {:.2f}\n'.\
      format(alpha, iters, theta, J(X,y,theta).item()))

print('Training the model... ')
# this actually trains our model and finds the optimal theta value
J_history, theta_min = gradient(X, y, alpha, theta, iters)
print('Done.')

print('\nFinal theta: {}\nFinal J: {:.2f}'.format(theta_min.T, J(X,y,theta_min.T).item()))

'''
Now that we have the model trained, we can use it to predict the profit/loss

Usually, since we want to solve a real problem, we define our function to accept real numbers, not rescaled ones.
However, we have to remember, that the model itself is trained on rescaled data, so we have to provide it.
'''

# This function will calculate the predicted profit
def predict_profit(population):
    pop = population / 10000
    return [1, pop] * theta_min * 10000

# Now, let's check for a random city
p = 50000 + 100000 * np.random.random()
print('\n'+40*'=')
print('\nBased on learned data, predicted profit for a city of population of {:,.0f} is ${:,.2f}.\n'.\
      format(p, predict_profit(p).item()))

# For the business decision, it would also be good to know what is the minimal population of a city to start the
# profitable business (predicted value is at least positive)
p_min = -theta_min[0].item() / theta_min[1].item() * 10000
print('In order for the business to be profitable, it has to be started in a city with population larger than {:,.0f}.'\
      .format(p_min))
print('\n'+40*'=')
print('\nNOTE: The code initializes the model with different theta each time, thus the model predicts different minimal'
      'viable population at each runtime.')