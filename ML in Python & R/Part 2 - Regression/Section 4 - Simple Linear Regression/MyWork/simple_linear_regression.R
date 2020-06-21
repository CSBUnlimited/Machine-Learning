
### Data preprocessing

## Import dataset
dataset = read.csv('Salary_Data.csv')
# dataset = dataset[, 2:3]

## Split dataset to train and test

# Need a new package caTools
# install.packages('caTools')
library(caTools) #using this we can import caTools directly

# In python random_state, in R it is seed
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) # SplitRatio is train set ratio oposite of python, This returns True - decide to goto train/ False- goto test set

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# ## Feature scaling
# # Facot is not numeric
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])


## Fitting Simple linear regression to training data set
regressor = lm(formula = Salary ~ YearsExperience, # Salary ~ YearsExperience | means salary propotional to YearsExperience 
                data = training_set)
# In console type summary(regressor) to get a summary
# Coefficient
# *** stars mean highly statistical significant
# Pr Value (P value) - lower the value is more significant, less than 5% mean high impact

## Predict the Test set results
y_pred = predict(regressor, newdata = test_set)

## Visulalizing the trainging set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') + # aes use for separate x and y axises
  ggtitle('Salary vs Experience (Traning Set)') +
  xlab('Years of Experience') +
  ylab('Salary')


## Visulalizing the test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') + # No need to change this from previous, regressor already trained
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') +
  ylab('Salary')

