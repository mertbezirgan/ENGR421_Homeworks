import math
import matplotlib.pyplot as plt
import numpy as np


def decision_tree_reg(P, train_n, x_train, y_train):
    
  node_indices = {}
  is_terminal = {}
  need_split = {}
  node_splits = {}
  
  # put all training instances into the root node
  node_indices[1] = np.array(range(train_n))
  is_terminal[1] = False
  need_split[1] = True
  
  while True:
      
    split_nodes = [key for key, value in need_split.items() if value == True]
   
    if len(split_nodes) == 0:
      break
  
    for split_node in split_nodes:
      data_indices = node_indices[split_node]
      need_split[split_node] = False
      
      if len(data_indices) <= P:
        node_splits[split_node] = np.mean(y_train[data_indices])
        is_terminal[split_node] = True
      else:
        is_terminal[split_node] = False
        unique_values = np.sort(np.unique(x_train[data_indices]))
        split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
        split_scores = np.repeat(0.0, len(split_positions))
        
        for s in range(len(split_positions)):
          left_indices = data_indices[x_train[data_indices] <= split_positions[s]]
          right_indices = data_indices[x_train[data_indices] > split_positions[s]]
          error=0
          left_mean = np.mean(y_train[left_indices])
          right_mean = np.mean(y_train[right_indices])
          error = error + np.sum((y_train[left_indices]-left_mean)**2) + np.sum((y_train[right_indices] - right_mean) ** 2)
          split_scores[s] = error / (len(left_indices) + len(right_indices))
          
        best_splits = split_positions[np.argmin(split_scores)]
        node_splits[split_node] = best_splits
        
        left_indices = data_indices[x_train[data_indices] <= best_splits]
        node_indices[2 * split_node] = left_indices
        is_terminal[2 * split_node] = False
        need_split[2 * split_node] = True
        
        right_indices = data_indices[x_train[data_indices] > best_splits]
        node_indices[2 * split_node + 1] = right_indices
        is_terminal[2 * split_node + 1] = False
        need_split[2 * split_node + 1] = True
        
  return is_terminal, node_splits

def get_regression_results(data, node_splits, is_terminal):
    N = data.shape[0]
    regression_results = np.zeros(N)
    for i in range(N):
        tree_index = 1
        while True:
            if is_terminal[tree_index] == True:
                regression_results[i] = node_splits[tree_index]
                break
            else:
                if data[i] <= node_splits[tree_index]:
                    tree_index *= 2
                else:
                    tree_index *= 2
                    tree_index += 1
    return regression_results

def get_rmse(y_truth, y_pred):
    return np.sqrt(np.mean((y_truth - y_pred)**2))

data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header=True)

x_train = np.zeros(100)
y_train = np.zeros(100)
x_test = np.zeros(33)
y_test = np.zeros(33)

x_train[:] = data_set[:100,0]
y_train[:] = data_set[:100,1]
x_test[:] = data_set[100:134,0]
y_test[:] = data_set[100:134,1]

train_n = len(y_train)
is_terminal = decision_tree_reg(15, train_n, x_train, y_train)[0]
node_splits = decision_tree_reg(15, train_n, x_train, y_train)[1]

data_interval = np.linspace(0, 60, 500)
regression_results = get_regression_results(data_interval, node_splits, is_terminal)

plt.figure(figsize = (15, 5))
plt.plot(x_train,y_train,"b.", markersize = 10,label="Training")
plt.plot(x_test,y_test,"r.", markersize = 10,label="Test")
    
plt.plot(data_interval, regression_results, "k-")

plt.xlabel("x")
plt.ylabel("y")
plt.title("h=3")

plt.legend(loc='upper left')
plt.show()


test_results = get_regression_results(x_test, node_splits, is_terminal)
rmse = get_rmse(y_test, test_results)
print("RMSE is {:.4f} when P is 15".format(rmse))

rmse_values = np.zeros(10)
p_values = np.arange(5, 55, 5)

for i in range(p_values.shape[0]):
    is_terminal_iter = decision_tree_reg(p_values[i], train_n, x_train, y_train)[0]
    node_splits_iter = decision_tree_reg(p_values[i], train_n, x_train, y_train)[1]
    rmse_values[i] = get_rmse(y_test, get_regression_results(x_test, node_splits_iter, is_terminal_iter))

plt.figure(figsize = (15, 5))
plt.plot(p_values, rmse_values, "ko-", linewidth=2, markersize = 10)
plt.show()




























