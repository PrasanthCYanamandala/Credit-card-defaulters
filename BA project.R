install.packages(c("caret","xfun","randomForest","adabag","neuralnet","ggplot2","pROC","corrplot"))

# Load necessary libraries
library(caret)
library(randomForest)
library(adabag)
library(neuralnet)
library(ggplot2)
library(pROC)
library(corrplot)

# Assuming the dataset 'UCI_Credit_Card.csv' has been read into 'df'
df <- read.csv("UCI_Credit_Card.csv")

# Check for missing values 
# Replace NAs with the mean for numerical columns
numerical_columns <- sapply(df, is.numeric)
df[numerical_columns] <- lapply(df[numerical_columns], function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  return(x)
})

# Replace NAs with "new" for categorical columns
categorical_columns <- sapply(df, is.factor)
df[categorical_columns] <- lapply(df[categorical_columns], function(x) {
  levels(x) <- c(levels(x), "new")
  x[is.na(x)] <- "new"
  return(x)
})

# Clip outliers
df[numerical_columns] <- lapply(df[numerical_columns], function(x) {
  quantiles <- quantile(x, probs = c(0.01, 0.99), na.rm = TRUE)
  x[x < quantiles[1]] <- quantiles[1]
  x[x > quantiles[2]] <- quantiles[2]
  return(x)
})

# Rename the dependent variable column
colnames(df)[colnames(df) == "default.payment.next.month"] <- "DEFAULT"
df$DEFAULT <- as.numeric(df$DEFAULT)

# Change column types to factors
df$SEX <- as.factor(df$SEX)
df$EDUCATION <- as.factor(df$EDUCATION)
df$MARRIAGE <- as.factor(df$MARRIAGE)

# Plotting Correlation Matrix 
r = cor(df[-c(3, 4, 5, 26, 27)], method = 'pearson')
col_gd <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

corrplot(r, method = "color", col = col_gd(200),
         type = "upper", order = "hclust",
         addCoef.col = "Black",
         tl.col = "black", tl.srt = 45, number.cex = 0.5,tl.cex = 0.4)

# Scale only numeric columns in df excluding the 24th column
numeric_columns <- sapply(df[-24], is.numeric)
df[numeric_columns] <- lapply(df[numeric_columns], scale)

df$DEFAULT <- as.factor(df$DEFAULT)
df$PAY_0 = as.factor(df$PAY_0)

# Create a new variable "GENDER" based on the "SEX" column
df$GENDER <- ifelse(df$SEX == 1, "Male", "Female")

#Checking Defaulters by Gender to get a clear picture of our data.
ggplot(data = df, mapping = aes(x = SEX, fill = GENDER)) +
  geom_bar() +
  ggtitle("Gender") +
  stat_count(aes(label = ..count..), geom = "label")
ggplot(data = df, mapping = aes(x = EDUCATION, fill = GENDER)) +
  geom_bar() +
  ggtitle("EDUCATION") +
  stat_count(aes(label = ..count..), geom = "label")

# Splitting data into training and testing datasets
set.seed(123) # for reproducibility
train_index <- createDataPartition(df$DEFAULT, p = 0.7, list = FALSE)
trainset <- df[train_index, ]
testset <- df[-train_index, ]

# Convert factor variables to dummy variables
trainset$SEX <- as.numeric(as.factor(trainset$SEX))
trainset$PAY_0 <- as.numeric(as.factor(trainset$PAY_0))

#Similar conversions for the test set
testset$SEX <- as.numeric(as.factor(testset$SEX))
testset$PAY_0 <- as.numeric(as.factor(testset$PAY_0))


# Logistic Regression Model
logit.reg <- glm(DEFAULT ~ ., data = trainset, family = "binomial")
logit.reg.pred <- predict(logit.reg, testset, type = "response")
logit.reg.pred.classes <- ifelse(logit.reg.pred > 0.5, 1, 0)

# Random Forest Model
classifier.rf <- randomForest(DEFAULT ~ ., data = trainset, ntree = 50)
our.predict.rf <- predict(classifier.rf, newdata = testset)

# Boosting Model
boost <- boosting(DEFAULT ~ ., data = trainset, mfinal = 20)
pred.boost <- predict(boost, testset)

# Neural Networks
nn <- neuralnet(DEFAULT ~ PAY_0 + SEX + BILL_AMT1 + PAY_AMT1, data = trainset, linear.output = FALSE, hidden = 1, learningrate = 0.05)
nn.pred <- compute(nn, testset[, c("PAY_0", "SEX", "BILL_AMT1", "PAY_AMT1")])
nn.pred.classes <- ifelse(nn.pred$net.result > 0.5, 1, 0)
# Model Evaluation
# Assuming the confusionMatrix function is available, otherwise use table() to create confusion matrix'''
# Confusion matrix for Random Forest Model
rf_confusion <- confusionMatrix(as.factor(our.predict.rf), as.factor(testset$DEFAULT))
rf_confusion
# Confusion matrix for Logistic Regression Model
confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(testset$DEFAULT))

lr_confusion <- confusionMatrix(as.factor(logit.reg.pred.classes), as.factor(testset$DEFAULT))
lr_confusion
# Confusion matrix for Boosting Model
boost_confusion <- confusionMatrix(as.factor(pred.boost$class), as.factor(testset$DEFAULT))
boost_confusion
# Confusion matrix for Neural Networks
confusionMatrix(as.factor(nn.pred.classes[,2]), as.factor(testset$DEFAULT))

# ROC Curve and AUC for each model
roc_rf <- roc(as.numeric(testset$DEFAULT), as.numeric(our.predict.rf))
roc_lr <- roc(as.numeric(testset$DEFAULT), as.numeric(logit.reg.pred))
roc_boost <- roc(as.numeric(testset$DEFAULT), as.numeric(pred.boost$class))
roc_nn <- roc(as.numeric(testset$DEFAULT), as.numeric(nn.pred.classes))

# Plotting ROC Curves 

# ROC Curve for Logistic Regression
roc_lr <- roc(response = testset$DEFAULT, predictor = as.numeric(logit.reg.pred))
plot(roc_lr, main="ROC for Logistic Regression")
auc(roc_lr)

# ROC Curve for Random Forest
roc_rf <- roc(response = testset$DEFAULT, predictor = as.numeric(our.predict.rf))
plot(roc_rf, main="ROC for Random Forest")
auc(roc_rf)

# ROC Curve for Boosting Model
roc_boost <- roc(response = testset$DEFAULT, predictor = as.numeric(pred.boost$class))
plot(roc_boost, main="ROC for Boosting Model")
auc(roc_boost)

# ROC Curve for Neural Network
# Assuming nn.pred.classes is a numeric vector of predictions
roc_nn <- roc(response = testset$DEFAULT, predictor = as.numeric(nn.pred.classes))
plot(roc_nn, main="ROC for Neural Network")
auc(roc_nn)

 