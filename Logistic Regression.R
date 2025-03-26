rm(list=ls()) ; gc()

#Logistic Regression
library(tidyverse)
library(readr)

  library(caret)
library(corrplot)
library(scales)
  library(dplyr)
library(mlbench)
  library(pROC)
  library(ROCR)
set.seed(1234)
  library(caret)
  library(readxl)

  library("dplyr")

data_1 <- read.csv("UCI_Credit_Card.csv")

df<- data_1 %>%
  select(-1)

#Shape of data
dim(data_1)

#We can change our column types to ensure that our algorithm does not mistake 
#Ordinal values for our nominal and Scale our values to better help our algorithm

df$SEX<- as.factor(df$SEX)
df$EDUCATION<- as.factor(df$EDUCATION)
df$MARRIAGE <- as.factor(df$MARRIAGE)


#Changing name of our target variable
colnames(df)[colnames((df)) == "default.payment.next.month"] = "DEFAULT"

#Logistic Regression 

set.seed(2)

#Validation dataset
test_data <- df 

#Considering an 80/20 split
train_split <- 0.8

#Creating the split
train_index <- createDataPartition(test_data$DEFAULT,p = train_split, list = FALSE,times = 1)

training_set <- test_data[train_index,]
test_dataset <- test_data[-train_index,]

#Fitting the training data to a logistic regression model
logit.reg <- glm(DEFAULT ~ ., data = training_set, family = "binomial")

summary(logit.reg)

## use predict() with type = "response" to compute predicted probabilities. 

logit.reg.pred <- predict(logit.reg, test_dataset, type = "response")

## Predicting first 10 records
data.frame(actual = test_dataset$DEFAULT[1:10], predicted = logit.reg.pred[1:10])


logit.reg.pred.classes <- ifelse(logit.reg.pred > 0.5, 1, 0)

#
confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(test_dataset$DEFAULT))


#ROC measures model's accuracy 

#The Receiver Operating Characteristics(ROC) is a measure of a classification model's performance 
#over various thresholds. 
#This is accomplished using the parameters of a Confusion Matrix. 
#The Area Under the Curve is a metric to evaulate the ROC of various classification models.

logis <- glm(DEFAULT ~ ., data=training_set, family=binomial)
roc(training_set$DEFAULT, as.vector(fitted.values(logis)), percent=T,   boot.n=1000, ci.alpha=0.9, stratified=FALSE, plot=TRUE, xlab="False Positive %", ylab="True Positive %")

#AUC
#AUC measures model's accuracy vs other models

logis2 <- glm(DEFAULT ~ LIMIT_BAL+SEX+EDUCATION+MARRIAGE+ PAY_0+ PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+ PAY_AMT1+PAY_AMT2,
              data=training_set, family=binomial)
roc(training_set$DEFAULT, as.vector(fitted.values(logis2)), percent=F,   boot.n=1000, ci.alpha=0.9, stratified=FALSE, plot=TRUE)

# Fit logistic regression model
logis <- glm(DEFAULT ~ ., data=training_set, family=binomial)

# Calculate the ROC curve
roc_obj <- roc(training_set$DEFAULT, as.vector(fitted(logis)), plot=TRUE)

# Calculate the AUC
auc_value <- auc(roc_obj)

# Print the AUC
print(paste("The AUC for the logistic regression model is:", auc_value))

