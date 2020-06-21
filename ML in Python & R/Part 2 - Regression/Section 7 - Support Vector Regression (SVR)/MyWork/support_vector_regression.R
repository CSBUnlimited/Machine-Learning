
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

## Fitting SVR to data set
# install.packages('e1071')

library(e1071) # this library do the feature scaling

regressor = svm(formula = Salary ~ ., 
                data = dataset,
                type = 'eps-regression') # We doing regression, if we do svm which is classification we need to choose classification 


## Visualize data
# install.packages('ggplot2')
library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
x_grid = data.frame(Level = x_grid)

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid$Level, y = predict(regressor, newdata = x_grid)), colour = 'blue') +
  ggtitle(label = 'Truth or Bluff (SVR)') +
  xlab(label = 'Level') +
  ylab(label = 'Salary')


## Predications
predit_data = data.frame(Level = 6.5)

y_pred = predict(regressor, newdata = predit_data)















