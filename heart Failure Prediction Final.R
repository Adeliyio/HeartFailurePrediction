#https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv

# Set the working directory
setwd("C:/Users/User/Desktop/Personal Projects/Heart Failure Prediction")


# Load necessary libraries
library(dplyr)
library(corrplot)
library(caret)
library(e1071)
library(randomForest)
library(rpart)
library(mlr)
library(kknn)
library(ROSE)
library(pROC)
library(xgboost)
library(Boruta)

# Read the CSV file into a variable called pd
pd <- read.csv("heart_failure_clinical_records_dataset.csv")

#check for missing data
sum(is.na(pd))
str(pd)

# Split the data into training and test sets
set.seed(123)  # Set seed for reproducibility
train_indices <- createDataPartition(pd$DEATH_EVENT, p = 0.7, list = FALSE)
train_data <- pd[train_indices, ]
test_data <- pd[-train_indices, ]

# Explore the data
summary(train_data)
str(train_data)
head(train_data,10)

# Plot the bar chart for Death events
barplot(table(train_data$DEATH_EVENT), xlab = "Death events", ylab = "Number of instances")

# Perform exploratory data analysis (EDA) and data preprocessing
numeric_vars <- c("age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_creatinine", "serum_sodium")
categorical_vars <- c("anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT")

# Convert categorical variables to factors
train_data[, categorical_vars] <- lapply(train_data[, categorical_vars], factor)
test_data[, categorical_vars] <- lapply(test_data[, categorical_vars], factor)

train_data$DEATH_EVENT <- factor(train_data$DEATH_EVENT, levels = c("0", "1"))
test_data$DEATH_EVENT <- factor(test_data$DEATH_EVENT, levels = c("0", "1"))

# Plot histograms for numeric variables
par(mfrow = c(2, 3))
for (var in numeric_vars) {
  hist(train_data[[var]], main = var, xlab = var)
}

# Calculate correlations
correlations <- cor(train_data[, numeric_vars])

# Plot correlation matrix
corrplot(correlations, method = "circle")

# Upsample the training data using SMOTE (Synthetic Minority Over-sampling Technique)
set.seed(1234)
train_upsampled <- ROSE(DEATH_EVENT ~ ., data = train_data, seed = 1234)$data

# Check the balance
table(train_upsampled$DEATH_EVENT)

# Feature selection using Boruta
set.seed(123)
boruta_data <- train_data[, -which(names(train_data) == "DEATH_EVENT")]  # Remove the target variable
boruta_target <- train_data$DEATH_EVENT
boruta_output <- Boruta(boruta_data, boruta_target)
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)

# Get the selected variables from Boruta and update train_data
selected_vars <- c("age", "ejection_fraction", "serum_creatinine", "serum_sodium", "time")
train_data <- train_data[, c(selected_vars, "DEATH_EVENT")]

# Build and evaluate the models
model_results <- data.frame(Model = character(), Accuracy = numeric(), Sensitivity = numeric(), Specificity = numeric(), stringsAsFactors = FALSE)

# Logistic Regression
model_logit <- glm(DEATH_EVENT ~ ., family = binomial(link = "logit"), data = train_data)
pred_logit <- ifelse(predict(model_logit, newdata = test_data[, selected_vars], type = "response") >= 0.5, 1, 0)

# Calculate evaluation metrics
accuracy_logit <- sum(pred_logit == test_data$DEATH_EVENT) / length(pred_logit)
sensitivity_logit <- sum(pred_logit[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_logit <- sum(pred_logit[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("Logistic Regression", accuracy_logit, sensitivity_logit, specificity_logit))

# Decision Tree
model_dt <- rpart(DEATH_EVENT ~ ., data = train_data, method = "class")
pred_dt <- ifelse(predict(model_dt, newdata = test_data[, selected_vars], type = "class") == "1", 1, 0)

# Calculate evaluation metrics
accuracy_dt <- sum(pred_dt == test_data$DEATH_EVENT) / length(pred_dt)
sensitivity_dt <- sum(pred_dt[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_dt <- sum(pred_dt[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("Decision Tree", accuracy_dt, sensitivity_dt, specificity_dt))

# Random Forest
model_rf <- randomForest(DEATH_EVENT ~ ., data = train_data, ntree = 500)
pred_rf <- ifelse(predict(model_rf, newdata = test_data[, selected_vars], type = "prob")[, 2] >= 0.3, 1, 0)

# Calculate evaluation metrics
accuracy_rf <- sum(pred_rf == test_data$DEATH_EVENT) / length(pred_rf)
sensitivity_rf <- sum(pred_rf[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_rf <- sum(pred_rf[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("Random Forest", accuracy_rf, sensitivity_rf, specificity_rf))

# Support Vector Machines (SVM)
model_svm <- svm(DEATH_EVENT ~ ., data = train_data)
pred_svm <- predict(model_svm, newdata = test_data[, selected_vars])

# Calculate evaluation metrics
accuracy_svm <- sum(pred_svm == test_data$DEATH_EVENT) / length(pred_svm)
sensitivity_svm <- sum(pred_svm[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_svm <- sum(pred_svm[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("Support Vector Machines", accuracy_svm, sensitivity_svm, specificity_svm))

# K-Nearest Neighbors (KNN)
model_knn <- kknn(DEATH_EVENT ~ ., train_data, test_data[, selected_vars], train_data$DEATH_EVENT, k = 13)
pred_knn <- ifelse(as.numeric(fitted(model_knn)) >= 0.5, 1, 0)

# Calculate evaluation metrics
accuracy_knn <- sum(pred_knn == test_data$DEATH_EVENT) / length(pred_knn)
sensitivity_knn <- sum(pred_knn[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_knn <- sum(pred_knn[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("K-Nearest Neighbors", accuracy_knn, sensitivity_knn, specificity_knn))

# Print the model results
print(model_results)

# ROC curves
# Logistic Regression ROC
roc_logit <- roc(test_data$DEATH_EVENT, predict(model_logit, newdata = test_data[, selected_vars], type = "response"))
plot(roc_logit, col = "blue", main = "ROC Curves")
legend("bottomright", legend = "Logistic Regression", col = "blue", lwd = 2)

# Decision Tree ROC
roc_dt <- roc(test_data$DEATH_EVENT, predict(model_dt, newdata = test_data[, selected_vars], type = "prob")[, 2])
plot(roc_dt, add = TRUE, col = "red")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("blue", "red"), lwd = 2)

# Random Forest ROC
roc_rf <- roc(test_data$DEATH_EVENT, predict(model_rf, newdata = test_data[, selected_vars], type = "prob")[, 2])
plot(roc_rf, add = TRUE, col = "green")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest"), col = c("blue", "red", "green"), lwd = 2)

# SVM ROC
roc_svm <- roc(test_data$DEATH_EVENT, as.numeric(pred_svm))
plot(roc_svm, add = TRUE, col = "orange")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest", "SVM"), col = c("blue", "red", "green", "orange"), lwd = 2)

# KNN ROC
roc_knn <- roc(test_data$DEATH_EVENT, as.numeric(pred_knn))
plot(roc_knn, add = TRUE, col = "purple")
legend("bottomright", legend = c("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"), col = c("blue", "red", "green", "orange", "purple"), lwd = 2)


# Gradient Boosting
# Encode DEATH_EVENT as a binary variable
train_data$DEATH_EVENT <- ifelse(train_data$DEATH_EVENT == "0", 0, 1)

# Convert train_data to matrix format for xgboost
train_matrix <- xgb.DMatrix(as.matrix(train_data[, selected_vars]), label = train_data$DEATH_EVENT)

# Convert test_data to matrix format for xgboost
test_matrix <- xgb.DMatrix(as.matrix(test_data[, selected_vars]))

# Define the parameters for gradient boosting
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the gradient boosting model
model_gb <- xgb.train(params, data = train_matrix, nrounds = 100)

# Make predictions using the gradient boosting model
pred_gb <- predict(model_gb, test_matrix)

# Convert predicted probabilities to class labels
pred_gb <- ifelse(pred_gb >= 0.3, 1, 0)

# Calculate evaluation metrics
accuracy_gb <- sum(pred_gb == test_data$DEATH_EVENT) / length(pred_gb)
sensitivity_gb <- sum(pred_gb[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_gb <- sum(pred_gb[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("Gradient Boosting", accuracy_gb, sensitivity_gb, specificity_gb))

# Naive Bayes
# Train the naive Bayes model
model_nb <- naiveBayes(DEATH_EVENT ~ ., data = train_data)

# Make predictions using the naive Bayes model
pred_nb <- predict(model_nb, newdata = test_data[, selected_vars])

# Calculate evaluation metrics
accuracy_nb <- sum(pred_nb == test_data$DEATH_EVENT) / length(pred_nb)
sensitivity_nb <- sum(pred_nb[test_data$DEATH_EVENT == 1] == 1) / sum(test_data$DEATH_EVENT == 1)
specificity_nb <- sum(pred_nb[test_data$DEATH_EVENT == 0] == 0) / sum(test_data$DEATH_EVENT == 0)

# Store the results
model_results <- rbind(model_results, c("Naive Bayes", accuracy_nb, sensitivity_nb, specificity_nb))

# Print the final model results
print(model_results)


# Further evaluate random forest and gradient boosting
# Convert predicted values to factors with the same levels
pred_rf <- factor(pred_rf, levels = c("0", "1"))
test_data$DEATH_EVENT <- factor(test_data$DEATH_EVENT, levels = c("0", "1"))

# Random Forest Evaluation
# Calculate additional evaluation metrics
conf_rf <- confusionMatrix(pred_rf, test_data$DEATH_EVENT, positive = "1")

precision_rf <- conf_rf$byClass[["Pos Pred Value"]]  # Precision
recall_rf <- conf_rf$byClass[["Sensitivity"]]  # Recall
f1_score_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)  # F1 Score

# Print the evaluation metrics
cat("Random Forest Evaluation:\n")
cat("Accuracy:", accuracy_rf, "\n")
cat("Sensitivity:", sensitivity_rf, "\n")
cat("Specificity:", specificity_rf, "\n")
cat("Precision:", precision_rf, "\n")
cat("F1 Score:", f1_score_rf, "\n\n")


# Convert predicted values to factors with the same levels
pred_gb <- factor(pred_gb, levels = c("0", "1"))
test_data$DEATH_EVENT <- factor(test_data$DEATH_EVENT, levels = c("0", "1"))


# Gradient Boosting Evaluation
# Calculate additional evaluation metrics
conf_gb <- confusionMatrix(pred_gb, test_data$DEATH_EVENT, positive = "1")

precision_gb <- conf_gb$byClass[["Pos Pred Value"]]  # Precision
recall_gb <- conf_gb$byClass[["Sensitivity"]]  # Recall
f1_score_gb <- 2 * (precision_gb * recall_gb) / (precision_gb + recall_gb)  # F1 Score

# Print the evaluation metrics
cat("Gradient Boosting Evaluation:\n")
cat("Accuracy:", accuracy_gb, "\n")
cat("Sensitivity:", sensitivity_gb, "\n")
cat("Specificity:", specificity_gb, "\n")
cat("Precision:", precision_gb, "\n")
cat("F1 Score:", f1_score_gb, "\n")
