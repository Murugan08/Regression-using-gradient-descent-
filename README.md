Introduction:
In computing, a compute kernel is a routine compiled for high throughput accelerators such as graphics processing units (GPUs),  separate from but used by a main program (typically running on a central processing unit). They are sometimes called compute shaders, sharing execution units with vertex shaders and pixel shaders on GPUs, but are not limited to execution on one class of device, or graphics
The project is to demonstrate linear and logistic regression to predict the GPU run time which is done based on 14 different configurations of processor such as local work group size, local memory shape etc. 
Dataset: 
Dataset can be downloaded from: https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance
This data set measures the running time of a matrix-matrix product A*B = C, where all matrices have size 2048 x 2048, using a parameterizable SGEMM GPU kernel with 241600 possible parameter combinations. For each tested combination, 4 runs were performed and their results are reported as the 4 last columns. All times are measured in milliseconds*.

There are 14 parameter, the first 10 are ordinal and can only take up to 4 different powers of two values, and the 4 last variables are binary. Out of 1327104 total parameter combinations, only 241600 are feasible (due to various kernel constraints). This data set contains the results for all these feasible combinations.

Task:
By using packages like Numpy, Pandas and various plotting packages, a gradient decent algorithm to calculate the cost function and to find the global minima by tuning the value of learning rate. By using this algorithm we estimate the optimal Beta values to construct the regression equation to predict the GPU computation time. 

Experimentation:
First experiment is to implement gradient descent algorithm in the dataset. I initially scaled the dataset using standard scalar from sklearn. Then the dataset has been split into test and train split with split percentage of 70 as training data and 30 as testing data.
GPU RUNTIME =𝛽0+𝛽1∗x1+𝛽2∗x2+𝛽3∗x3+𝛽4∗x4+𝛽5∗x5+ 𝛽6∗x6+𝛽7∗x7+𝛽8∗x8+𝛽9∗x9+𝛽10∗x10+𝛽11∗x11+𝛽12∗x12+𝛽13∗x13+𝛽14∗x14
I selected the following learning rates to run the algorithm 0.001,0.003,0.006,0.1. And the results were plotted as functions of alpha and cost of test and train of the samples taken from the dataset.

Part B
Now the gradient decent algorithm is implemented with the logistic regression model. By using gradient decent optimum cost is calculated. The sigmoid function was calculated by the formula 1 / (1 + np.exp(-x)) to calculate the gradient decent as to obtain the output between 0 or 1. Our current prediction function returns a probability score between 0 and 1. In order to map this to a discrete class, we select a threshold value or tipping point as the median of the runtime above which we will classify values into 1 or 0. 
