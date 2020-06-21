
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
dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4
regressor = lm(formula = Salary ~ ., # data = training_set) 
                   data = dataset)


## Visualize data
# install.packages('ggplot2')
library(ggplot2)

x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
x_grid = data.frame(Level = x_grid)
x_grid$Level2 = x_grid$Level ^ 2
x_grid$Level3 = x_grid$Level ^ 3
x_grid$Level4 = x_grid$Level ^ 4

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid$Level, y = predict(pol_regressor, newdata = x_grid)), colour = 'blue') +
  ggtitle(label = 'Truth or Bluff (Regression Model)') +
  xlab(label = 'Level') +
  ylab(label = 'Salary')


## Predications
predit_data = data.frame(Level = 6.5)

predit_data$Level2 = predit_data$Level ^ 2
predit_data$Level3 = predit_data$Level ^ 3
predit_data$Level4 = predit_data$Level ^ 4

y_pred = predict(regressor, newdata = predit_data)















