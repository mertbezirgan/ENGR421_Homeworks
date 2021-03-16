# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:19:10 2020

@author: Mert
"""

import math
import matplotlib.pyplot as plt
import numpy as np

data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header=True)

x_train = np.zeros(100)
y_train = np.zeros(100)
x_test = np.zeros(33)
y_test = np.zeros(33)

x_train[:] = data_set[:100,0]
y_train[:] = data_set[:100,1]
x_test[:] = data_set[100:133,0]
y_test[:] = data_set[100:133,1]

min_val = min(x_train)
max_val = max(x_train)

# get number of classes and number of samples
K = np.max(y_train)
N = data_set.shape[0]

bin_width = 3
origin = 0

left_borders = np.arange(min_val, max_val, bin_width)
right_borders = np.arange(min_val + bin_width, max_val + bin_width, bin_width)

#regressogram

regressogram_results = np.zeros(len(left_borders))

for b in range(len(left_borders)):
    regressogram_results[b]= np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b]))*y_train) / np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b]))
    

plt.figure(figsize = (15, 5))
plt.plot(x_train,y_train,"b.", markersize = 10,label="Training")
plt.plot(x_test,y_test,"r.", markersize = 10,label="Test")
    
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram_results[b], regressogram_results[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram_results[b], regressogram_results[b + 1]], "k-")   


plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")

plt.legend(loc='upper left')
plt.show()

def get_bin_index(x):
    res = math.ceil((x-min_val) / bin_width)
    print(res)
    return res

#rmse for regressogram
rmse_reg = 0
for i in range(len(right_borders)):
    for j in range(len(y_test)):
        if x_test[j] <= right_borders[i] and left_borders[i] <= x_test[j]:
            rmse_reg = rmse_reg + (y_test[j] - regressogram_results[i])**2

rmse_reg = np.sqrt(rmse_reg/33)
print("Regressogram => RMSE is {:.4f} when h is 3".format(rmse_reg))



#Running mean smoother
rms_results = np.zeros(601)
data_interval = np.linspace(0, 60, 601)
counter = 0
for x in data_interval:
    rms_results[counter] = np.sum((((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))*y_train) / np.sum(((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width)))
    counter += 1


plt.figure(figsize = (15, 5))
plt.plot(x_train,y_train,"b.", markersize = 10,label="Training")
plt.plot(x_test,y_test,"r.", markersize = 10,label="Test")
    
plt.plot(data_interval, rms_results, "k-")


plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")

plt.legend(loc='upper left')
plt.show()


#rmse for running mean smoother
rmse_rms = 0
for i in range(len(x_test)):
    rmse_rms += (y_test[i] - rms_results[np.where(np.isclose(data_interval, x_test[i]))])**2

rmse_rms = np.sqrt(rmse_rms/33)
print("Running Mean Smoother => RMSE is {:.4f} when h is 3".format(rmse_rms[0]))


#Kernel smoother
bin_width = 1

def k(u):
    return np.exp(-u**2/2)/np.sqrt(2*math.pi)

kernel_res = np.zeros(len(data_interval))

for i in range(len(data_interval)):
    kernel_res[i] = np.sum(k((data_interval[i] - x_train)/bin_width )*y_train) / np.sum(k((data_interval[i] - x_train)/bin_width ))

plt.figure(figsize = (15, 5))
plt.plot(x_train,y_train,"b.", markersize = 10,label="Training")
plt.plot(x_test,y_test,"r.", markersize = 10,label="Test")
    
plt.plot(data_interval, kernel_res, "k-")


plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")

plt.legend(loc='upper left')
plt.show()


#RMSE for kernel smoother

rmse_kernel = 0
for i in range(len(x_test)):
    rmse_kernel += (y_test[i] - kernel_res[np.where(np.isclose(data_interval, x_test[i]))])**2

rmse_kernel = np.sqrt(rmse_kernel/33)
print("Kernel Smoother => RMSE is {:.4f} when h is 1".format(rmse_kernel[0]))









    