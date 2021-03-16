import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = np.genfromtxt("hw03_data_set_images.csv", delimiter = ",")
labels = np.genfromtxt("hw03_data_set_labels.csv", dtype=str)
for i in range(labels.shape[0]):
    labels[i] = labels[i].split('"')[1] 


train_data = np.zeros((25*5,320))
test_data = np.zeros((14*5,320))
train_labels = np.chararray(25*5)
test_labels = np.chararray(14*5)

for i in range(0,5):
    train_data[25*i:(25+25*i),:] = data[(39*i):(25+39*i),:]
    test_data[14*i:(14+14*i),:] = data[(25+i*39):39*(i+1),:]
    train_labels[25*i:(25+25*i)] = labels[(39*i):(25+39*i)]
    test_labels[14*i:(14+14*i)] = labels[(25+i*39):39*(i+1)]


# im1 = np.transpose(train_data[0].reshape((16,20)))
# plt.imshow(im1)
    
def gen_y_truth(y):
    val = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        if y[i].decode("utf-8") == 'A':
            val[i] = 1
        elif y[i].decode("utf-8") == 'B':
            val[i] = 2
        elif y[i].decode("utf-8") == 'C':
            val[i] = 3
        elif y[i].decode("utf-8") == 'D':
            val[i] = 4
        elif y[i].decode("utf-8") == 'E':
            val[i] = 5
    return val

y_train_truth = gen_y_truth(train_labels)
y_test_truth = gen_y_truth(test_labels)


sample_means = np.array([np.sum(train_data[y_train_truth == (c + 1)], axis=0) for c in range(5)]) / 25
print(sample_means[0])
print(sample_means[1])
print(sample_means[2])
print(sample_means[3])
print(sample_means[4])

class_priors = [np.mean(y_train_truth == (c + 1)) for c in range(5)]

for i in range(5):
    plt.figure()
    plt.imshow(np.transpose(sample_means[i].reshape(16,20)), cmap="Greys")
    plt.xticks([])
    plt.yticks([])

def safe_log(x):
    return(np.log(x + 1e-100))

def calc_score(x):
    scores = np.zeros(5)
    for i in range(5):
        scores[i] = scores[i] + safe_log(class_priors[i])
        scores[i] = scores[i] + np.sum( x*safe_log(sample_means[i]) + ( (np.ones(320) - x)*safe_log(np.ones(320) - sample_means[i])) )
    return scores

train_predictions = np.zeros(125)
for i in range(125):
    train_predictions[i] = np.argmax(calc_score(train_data[i])) + 1

confusion_matrix = pd.crosstab(train_predictions, y_train_truth, rownames = ['y_pedicted'], colnames = ['y_test'])
print("training performance")
print(confusion_matrix)


test_predictions = np.zeros(70)
for i in range(70):
    test_predictions[i] = np.argmax(calc_score(test_data[i])) + 1


test_confusion_matrix = pd.crosstab(test_predictions, y_test_truth, rownames = ['y_pedicted'], colnames = ['y_test'])
print("test performance")
print(test_confusion_matrix)











