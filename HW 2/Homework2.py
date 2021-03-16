#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


data = np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")
labels = np.genfromtxt("hw02_data_set_labels.csv", dtype=str)
for i in range(labels.shape[0]):
    labels[i] = labels[i].split('"')[1] 


# In[3]:


# arr[0:6] first 6  ------ 6 test 4 train total 10
# arr[6:10] 7 8 9 10
# arr[10:16] 10 to 16
# arr[16:20] 17 to 20
train_data = np.zeros((25*5,320))
test_data = np.zeros((14*5,320))
train_labels = np.chararray(25*5)
test_labels = np.chararray(14*5)

for i in range(0,5):
    train_data[25*i:(25+25*i),:] = data[(39*i):(25+39*i),:]
    test_data[14*i:(14+14*i),:] = data[(25+i*39):39*(i+1),:]
    train_labels[25*i:(25+25*i)] = labels[(39*i):(25+39*i)]
    test_labels[14*i:(14+14*i)] = labels[(25+i*39):39*(i+1)]


# In[4]:


def sigmoid(W, x, wo):
    return 1/(1 + np.exp(-(np.matmul(x, W) + w0)))


# In[5]:


W = np.random.uniform(low = -0.01, high = 0.01, size = (320, 5))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, 5))
eta = 0.01
epsilon = 1e-3


# In[6]:


def gradient_W(X, y_truth, y_predicted):
    return(np.asarray([-np.sum(np.repeat((y_truth[:,c] - y_predicted[:,c])[:, None], X.shape[1], axis = 1) * X, axis = 0) for c in range(5)]).transpose())

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(y_truth - y_predicted, axis = 0))


# In[7]:


def gen_y_truth(y):
    val = np.zeros((y.shape[0], 5))
    for i in range(y.shape[0]):
        if y[i].decode("utf-8") == 'A':
            val[i][0] = 1
        elif y[i].decode("utf-8") == 'B':
            val[i][1] = 1
        elif y[i].decode("utf-8") == 'C':
            val[i][2] = 1
        elif y[i].decode("utf-8") == 'D':
            val[i][3] = 1
        elif y[i].decode("utf-8") == 'E':
            val[i][4] = 1
    return val


# In[8]:


iteration = 1
objective_values = []
y_truth = gen_y_truth(train_labels)

while 1:
    y_predicted = sigmoid(W, train_data, w0)
    objective_values = np.append(objective_values, 1/2*np.sum((y_truth-y_predicted)**2))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(train_data, y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth, y_predicted)

    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    iteration = iteration + 1


# In[9]:


plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.title("Training Error vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# In[10]:


y_predicted_classes = np.argmax(y_predicted, axis = 1) + 1
y_truth_classes = np.argmax(y_truth, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted_classes, y_truth_classes, rownames = ['y_pred'], colnames = ['y_truth'])
print("Training confusion matrix")
print(confusion_matrix)


# In[11]:


#testing
learned_w = W
learned_w0 = w0


# In[12]:


test_results = sigmoid(learned_w, test_data, learned_w0)
test_truth = gen_y_truth(test_labels)

test_results_classes = np.argmax(test_results, axis = 1) + 1
test_truth_classes = np.argmax(test_truth, axis = 1) + 1

confusion_matrix = pd.crosstab(test_results_classes, test_truth_classes, rownames = ['y_pedicted'], colnames = ['y_test'])
print("Testing confusion matrix")
print(confusion_matrix)

