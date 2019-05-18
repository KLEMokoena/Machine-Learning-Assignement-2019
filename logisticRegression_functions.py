import numpy as np
from math import log,exp

def trainlogisticR(xMatrix,yvector):
    m = len(xMatrix)
    theta_list = [] #stores the list of parameters for the logistic models
    for i in range(0,3):
        theta = np.ones((m,1))
        df = dcostfunction(xMatrix,yvector[i],theta)
        itercount = 0
        while itercount < 500:
            #updating all all theta simultaneously
            theta = theta - (0.3)*df
            itercount += 1
            df = dcostfunction(xMatrix, yvector[i], theta)
        theta_list.append(theta)
    return theta_list

def costfunction(xvector,yvector,theta):
    fx = 0
    for i in xvector:
        fx = fx + yvector[i]*log(hypothesisfunction(theta,xvector[i]),2)+(1-yvector[i])*log(1-hypothesisfunction(theta,xvector[i]))
    return fx

def dcostfunction(xMatrix,curry,theta): # derivative of the cost function
    df = np.zeros((len(xMatrix),1))
    for i in range(0,3):
        currX = xMatrix[:,i]
        df = df + (hypothesisfunction(theta,xMatrix[:,i]) - curry[i])*np.transpose([currX])
    return df


def hypothesisfunction(theta,xi): # this is the hypothesis function
    a = -1*np.dot(np.transpose(theta),xi)
    hypothesis = 1/(1+exp(a)) #logistic equation
    return hypothesis