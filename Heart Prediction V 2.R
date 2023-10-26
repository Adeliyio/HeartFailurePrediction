# Load libraries
library(tidyverse)
library(caret)
library(ROSE)
library(Boruta)
library(rpart)
library(rpart.plot)
library(ranger)
library(e1071)
library(mlr)
library(class)

# Set working directory
setwd("C:/Users/User/Desktop/Personal Projects/Heart Failure Prediction")

# Read the CSV file into a data frame called pd
pd <- read.csv("heart_failure_clinical_records_dataset.csv")

# Split the data into training and testing datasets
set.seed(123)
train_indices <- createDataPartition(pd$DEATH_EVENT, p = 0.7, list = FALSE)
train_data <- pd[train_indices, ]
test_data <- pd[-train_indices, ]

# Explore the data
summary(pd)
barplot(table(pd$DEATH_EVENT), xlab = "Death events", ylab = "Number of instances")

# Select numeric variables
numeric_vars <- c("age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium")
data_heart_numeric <- pd %>% select(all_of(numeric_vars))

# Plot histograms for numeric variables
data_heart_numeric %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

# Convert categorical variables to factors
categorical_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT")
train_data[categorical_vars] <- lapply(train_data[categorical_vars], factor)
test_data[categorical_vars] <- lapply(test_data[categorical_vars], factor)

# Remove the 'time' column from the data frames
train_data <- select(train_data, -time)
test_data <- select(test_data, -time)

# Upsample the training data
set.seed(1234)
up_heart_train <- ROSE(DEATH_EVENT ~ ., data = train_data, seed = 1234)$data

# Perform feature selection using Boruta
boruta_output <- Boruta(DEATH_EVENT ~ ., data = na.omit(train_data), doTrace = 0)
selected_vars <- getSelectedAttributes(boruta_output, withTentative = TRUE)

# Build logistic regression models
model_logit <- glm(DEATH_EVENT ~ ., family = binomial(link = "logit"), data = train_data[selected_vars])
model_logit_upsampling <- glm(DEATH_EVENT ~ ., family = binomial(link = "logit"), data = up_heart_train[selected_vars])

# Make predictions using logistic regression models
predict_logit <- predict(model_logit, newdata = test_data, type = "response")
predict_logit_up <- predict(model_logit_upsampling, newdata = test_data, type = "response")

# Encode DEATH_EVENT as a binary variable
test_data$DEATH_EVENT <- ifelse(test_data$DEATH_EVENT == "0", 0, 1)

# Convert predicted values to factor
predict_logit <- as.factor(predict_logit >= 0.5)
predict_logit_up <- as.factor(predict_logit_up >= 0.5)

# Calculate confusion matrices
confusionMatrix_logit <- confusionMatrix(predict_logit, test_data$DEATH_EVENT, positive = "1")
confusionMatrix_logit_up <- confusionMatrix(predict_logit_up, test_data$DEATH_EVENT, positive = "1")

# Decision Tree model
cart <- rpart(DEATH_EVENT ~ ., data = train_data, method = "class")
cart_up <- rpart(DEATH_EVENT ~ ., data = up_heart_train, method = "class")

# Make predictions using Decision Tree models
pred_dt <- predict(cart, newdata = test_data, type = "class")
pred_dt_up <- predict(cart_up, newdata = test_data, type = "class")

# Calculate confusion matrices
confusionMatrix_dt <- confusionMatrix(pred_dt, test_data$DEATH_EVENT, positive = "1")
confusionMatrix_dt_up <- confusionMatrix(pred_dt_up, test_data$DEATH_EVENT, positive = "1")

# Random Forest model
model_rf <- ranger(DEATH_EVENT ~ ., data = train_data, mtry = 2, num.trees = 500)
model_rf_upsampling <- ranger(DEATH_EVENT ~ ., data = up_heart_train, mtry = 2, num.trees = 500)

# Make predictions using Random Forest models
pred_rf <- predict(model_rf, data = test_data)$predictions
pred_rf_upsampling <- predict(model_rf_upsampling, data = test_data)$predictions

# Calculate confusion matrices
confusionMatrix_rf <- confusionMatrix(pred_rf, test_data$DEATH_EVENT, positive = "1")
confusionMatrix_rf_upsampling <- confusionMatrix(pred_rf_upsampling, test_data$DEATH_EVENT, positive = "1")

# Support Vector Machine (SVM) model
task_train <- makeClassifTask(data = train_data, target = "DEATH_EVENT")
task_test <- makeClassifTask(data = test_data, target = "DEATH_EVENT")

lrn_svm <- makeLearner("classif.svm", predict.type = "prob")
svm <- train(lrn_svm, task_train)

lrn_svm_up <- makeLearner("classif.svm", predict.type = "prob")
svm_up <- train(lrn_svm_up, task_train_up)

# Make predictions using SVM models
pred_svm <- predict(svm, task_test)
pred_svm_up <- predict(svm_up, task_test_up)

# Calculate confusion matrices
confusionMatrix_svm <- confusionMatrix(pred_svm, test_data$DEATH_EVENT, positive = "1")
confusionMatrix_svm_up <- confusionMatrix(pred_svm_up, test_data$DEATH_EVENT, positive = "1")

# K-Nearest Neighbors (KNN) model
knn13 <- knn(train = train_data[, numeric_vars], test = test_data[, numeric_vars], cl = train_data$DEATH_EVENT, k = 13)

# Calculate confusion matrix
confusionMatrix_knn13 <- confusionMatrix(knn13, test_data$DEATH_EVENT, positive = "1")

# Compile metrics for evaluation
metrics <- rbind(
  confusionMatrix_logit$overall[1],
  confusionMatrix_logit_up$overall[1],
  confusionMatrix_dt$overall[1],
  confusionMatrix_dt_up$overall[1],
  confusionMatrix_rf$overall[1],
  confusionMatrix_rf_upsampling$overall[1],
  confusionMatrix_svm$overall[1],
  confusionMatrix_svm_up$overall[1],
  confusionMatrix_knn13$overall[1]
)
rownames(metrics) <- c("Logit", "Logit (up-sampling)", "Decision Tree", "Decision Tree (up-sampling)",
                       "Random Forest", "Random Forest (up-sampling)", "SVM", "SVM (up-sampling)", "KNN13")
metrics


# ROC curves
roc_curve <- function(predictions, true_labels) {
  roc_data <- roc(true_labels, predictions)
  auc <- auc(roc_data)
  plot(roc_data, main = paste("ROC Curve (AUC =", auc, ")"), print.auc = TRUE)
}

roc_curve(predict_logit, test_data$DEATH_EVENT)
roc_curve(predict_logit_up, test_data$DEATH_EVENT)
roc_curve(as.numeric(pred_dt), test_data$DEATH_EVENT)
roc_curve(as.numeric(pred_dt_up), test_data$DEATH_EVENT)
roc_curve(as.numeric(pred_rf), test_data$DEATH_EVENT)
roc_curve(as.numeric(pred_rf_upsampling), test_data$DEATH_EVENT)
roc_curve(as.numeric(pred_svm), test_data$DEATH_EVENT)
roc_curve(as.numeric(pred_svm_up), test_data$DEATH_EVENT)
roc_curve(as.numeric(knn13), test_data$DEATH_EVENT)
