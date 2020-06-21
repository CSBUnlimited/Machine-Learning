
### Data preprocessing

## Import dataset
dataset = read.csv('50_Startups.csv')
# dataset = dataset[, 2:3]

## Encoding categorical data
# R see Categories as Facors so no need three columns

dataset$State = factor(x = dataset$State, 
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))

## Split dataset to train and test

# Need a new package caTools
# install.packages('caTools')

library(caTools) #using this we can import caTools directly

# In python random_state, in R it is seed
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8) # SplitRatio is train set ratio oposite of python, This returns True - decide to goto train/ False- goto test set

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# ## Feature scaling
# # Facot is not numeric
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

## Fitting Muliple Linear Regression to training set
regressor = lm(formula = Profit ~ .,
               data = training_set) # Instead of Profit ~ R.D.Spend + Administration + Marketing.Spend + State we can use Profit ~ . <- means all
 
## Predicting the test set result
y_pred = predict(regressor, newdata = test_set)

## Building optimal model using Backward Elimination

# State 2, State 3 means dummy variables. R manages it.
# Summary(resressor) in console to get summary of regressor
# look at significat (P Value) take significate

# Fitting Muliple Linear Regression to training set with indipendant variables
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = training_set)
summary(regressor)

# State 2 has highest P-value
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = training_set)
summary(regressor)

# Administration has highest P-value
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = training_set)
summary(regressor)

# Remove  Marketing.Spend
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
summary(regressor)













