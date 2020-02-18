import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as Sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier


dataset = pd.read_csv('sgemm_product.csv')


dataset.isnull().sum()


corrmat = dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(30,23))
g=Sb.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="YlOrBr")

columns = ['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']
df = dataset[columns].to_numpy()
a= df.mean(axis = 1)
#data.head()
datax = dataset.drop(['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'],axis = 1)
#datax.head()
datax['runtime'] = a
datax.head()


#dataset.rename(columns = {'Run1 (ms)':'run1', 'Run2 (ms)':'run2','Run3 (ms)':'run3', 'Run4 (ms)':'run4'}, inplace = True)
dataset
#Q1 = dataset.tail()
#Q3 = dataset.quantile(0.90)
#IQR = Q3-Q1
#print(IQR)
#columns = ['run1', 'run2', 'run3', 'run4']
#df = dataset[columns].to_numpy()
#runtime= df.mean(axis = 1)
#dataset.append(runtime)
#runtime= pd.DataFrame(runtime)
#data.head()
#datax = dataset.drop(['run1', 'run2', 'run3', 'run4'],axis = 1)

#datax.describe


datax.head()
#####################################################################
#Scale data
scaler= StandardScaler()
datax = scaler.fit_transform(datax)
datax = pd.DataFrame(datax)

datax


X = datax.iloc[:,0:13]
ones = np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)
X
y = datax.iloc[:,14:].values #.values converts it from pandas.coreX = dataset_LinReg.iloc[:,1:16]
ones = np.ones([X.shape[0],1])

y
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.030,random_state = 0)
theta = np.zeros([1,14])
# computecost
def computeCost(X_train,y_train,theta):
    tobesummed = np.power(((X_train @ theta.T)-y_train),2)
    return np.sum(tobesummed)/(2 * len(X_train))

# implement gradient descent
def gradientDescent(X_train,y_train,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X_train)) * np.sum(X_train * (X_train @ theta.T - y_train), axis=0)
        cost[i] = computeCost(X_train, y_train, theta)
    
    return theta,cost

# compute cost with all Betas as zeroes
finalCost_before_GD = computeCost(X_train,y_train,theta)
print(finalCost_before_GD)

#set hyper parameters
alpha = [0.001,0.003,0.006,0.1] #learning rate
iters = 500 #no.of iterations
threshold = [0.00001,0.000015,0.00002,0.000025]

finalCost_train = []
finalCost_test = []
for i in alpha:
    g,cost = gradientDescent(X_train,y_train,theta,iters,i)
    finalCost_train.append(cost[-1]) 
    finalCost_test.append(computeCost(X_test,y_test,g))

fig , ax = plt.subplots()
ax.plot(finalCost_train,alpha)
ax.set_xlabel('Cost')  
ax.set_ylabel('Alpha')  
ax.set_title('cost vs. Training set Alpha')  

fig , ax = plt.subplots()
ax.plot(finalCost_test,alpha)
ax.set_xlabel('Cost')  
ax.set_ylabel('Alpha')  
ax.set_title('cost vs. Test set Alpha')  

#################################################################################

learning_rate = [0.001,0.003,0.006,0.1]
max_iters_log = 500
LogCost_train = []
LogCost_test = []
accuracy = []

dataset = pd.DataFrame(dataset)






X_LogReg = dataset.iloc[:,0:14].values    
Y_LogReg = dataset.iloc[:,14:15].values


# Assigning 0s and 1s based on Median Average Runtime
Y_LogReg = np.where(Y_LogReg<= 69.79,0,1)
Y_LogReg = pd.DataFrame(Y_LogReg)


# Feature scaling
from sklearn.preprocessing import StandardScaler
dataset_sc_LogReg = StandardScaler()
X_LogReg = dataset_sc_LogReg.fit_transform(X_LogReg)
X_LogReg = pd.DataFrame(X_LogReg)
theta_log = np.zeros((X_LogReg.shape[1], 1))

X_train_LogReg,X_test_LogReg,Y_train_LogReg,Y_test_LogReg = train_test_split(X_LogReg,Y_LogReg,test_size = 0.03,random_state = 0)


def sigmoidfunction(x):
    return 1 / (1 + np.exp(-x))

def calculate_cost(X, y, theta):
    m = len(y)
    h = sigmoidfunction(X @ theta)
    epsilon = 1e-5
    cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        theta = theta - (learning_rate/m) * (X.T @ (sigmoidfunction(X @ theta) - y)) 
        cost_history[i] = calculate_cost(X, y, theta)

    return (cost_history, theta)

def pred_values(beta, X):
    
    pred_prob = sigmoidfunction(X.dot(beta))
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)


#initial_cost = calculate_cost(X_train_LogReg, Y_train_LogReg, theta_log)
#print("Initial Cost is: {} \n".format(initial_cost))

for alpha in learning_rate:
    (LogCost_history,theta_new) = gradient_descent(X_train_LogReg, Y_train_LogReg, theta_log, alpha, max_iters_log)
    LogCost_train.append(LogCost_history[-1]) 
    LogCost_test.append(calculate_cost(X_test_LogReg,Y_test_LogReg,theta_new))
    y_pred = pred_values(theta_new, X_test_LogReg)
    accuracy.append(accuracy_score(Y_test_LogReg,y_pred))
    
print(LogCost_test)
print(learning_rate)
print(accuracy)
LogCost_test[0]
print(LogCost_test.shape)

fig , ax = plt.subplots()
ax.plot(LogCost_train,learning_rate)
ax.set_xlabel('Cost')  
ax.set_ylabel('Learning Rate')  
ax.set_title('Cost vs. Training set Learning Rate')  

fig , ax = plt.subplots()
ax.plot(accuracy,learning_rate)
ax.set_xlabel('Accuracy')  
ax.set_ylabel('Learning Rate')  
ax.set_title('Accuracy vs. Test set Learning Rate')





