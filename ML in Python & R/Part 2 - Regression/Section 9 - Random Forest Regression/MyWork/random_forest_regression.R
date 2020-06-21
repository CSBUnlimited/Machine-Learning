
### Data preprocessing

## Import dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# We are not going to slitt dataset due to small number of observasions
# ## Split dataset to train and test
# 
# # Need a new package caTools
# # install.packages('caTools')
# 
# library(caTools) #using this we can import caTools directly
# 
# # In python random_state, in R it is seed
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8) # SplitRatio is train set ratio oposite of python, This returns True - decide to goto train/ False- goto test set
# 
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)


# ## Feature scaling
# # Facot is not numeric
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

## Fitting Regression to training set
# install.packages('randomForest')
library(randomForest)  















