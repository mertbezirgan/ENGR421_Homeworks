
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance
from scipy import stats


X_train = np.genfromtxt("hw07_training_images.csv",delimiter=",")
X_test = np.genfromtxt("hw07_test_images.csv",delimiter=",")
Y_train = np.genfromtxt("hw07_training_labels.csv",delimiter=",")
Y_test = np.genfromtxt("hw07_test_labels.csv",delimiter=",")


N = len(Y_train)
D = X_train.shape[1]
K = int(max(Y_train))


class_means = []

for i in range(K):
    class_means.append( np.mean(X_train[Y_train == i + 1,] , axis=0))
    
class_means = np.array(class_means)

X_train_minus_mean = []

for i in range(N):
    X_train_minus_mean.append( X_train[i, :] - class_means[np.int(Y_train[i]) - 1, :] )
    
X_train_minus_mean= np.array(X_train_minus_mean)

t_mean = np.mean(class_means, axis = 0)



def within_class_scatter():
    ret = np.zeros((D,D))
    class_covariances = [(np.dot(np.transpose(X_train[Y_train == (c + 1)] - class_means[c]), (X_train[Y_train == (c + 1)] - class_means[c]))) for c in range(K)]
    ret = class_covariances[0] + class_covariances[1] + class_covariances[2]
    return ret
        
def between_class_scatter():
    ret = np.zeros((D,D))
    for i in range(K):
        X_c = X_train[Y_train == i+1]
        mean_c = np.mean(X_c, axis = 0)
        n_c = X_c.shape[0]
        mean_d = (mean_c - t_mean).reshape(D,1)
        ret += n_c * np.dot(mean_d, np.transpose(mean_d))
    return ret
        
within_class_scatter_mat = within_class_scatter()
between_class_scatter_mat = between_class_scatter()

for d in range(D):
    within_class_scatter_mat[d,d] = within_class_scatter_mat[d,d] + 1e-10
    
    
#eigen values and eigen vectors
within_scatter_inversed = np.linalg.inv(within_class_scatter_mat)
values, vectors = la.eigh(np.dot(within_scatter_inversed, between_class_scatter_mat))

two_vectors = vectors[:, 0:2]
Z_train = np.dot(X_train, two_vectors)
Z_test = np.dot(X_test, two_vectors)



point_colors = ["#fc051a", "#004cff", "#00d150"]

plt.figure()
plt.title("training points")
for i in range(N):
    plt.scatter(Z_train[i,0], -Z_train[i,1], color=point_colors[np.int(Y_train[i])-1], s=5)
plt.show()


plt.figure()
plt.title("tests points")
for i in range(len(Y_test)):
    plt.scatter(Z_test[i,0],  -Z_test[i,1], color=point_colors[np.int(Y_test[i])-1], s=5)
plt.show()



train_predictions = []

for i in range(len(Z_train[:,1])):
    v = Z_train[i, :]
    initial_distances = np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        initial_distances[j] = distance.euclidean(v, Z_train[j, :])
    smallest_dists_indices = np.argsort(initial_distances)[:5]
    temp_labels = []
    for x in smallest_dists_indices:
        temp_labels.append(Y_train[x])
    prediction= stats.mode(temp_labels)[0]
    train_predictions.append(prediction)
    
print(np.transpose(np.array(confusion_matrix(train_predictions, Y_train))))



test_predictions = []

for i in range(len(Z_test[:,1])):
    v = Z_test[i, :]
    initial_distances = np.zeros(Z_train.shape[0])
    for j in range(len(Z_train[:,1])):
        initial_distances[j] = distance.euclidean(v, Z_train[j, :])
    smallest_dists_indices = np.argsort(initial_distances)[:5]
    temp_labels = []
    for x in smallest_dists_indices:
        temp_labels.append(Y_train[x])
    prediction= stats.mode(temp_labels)[0]
    test_predictions.append(prediction)
    
print(np.transpose(np.array(confusion_matrix(test_predictions, Y_test))))



