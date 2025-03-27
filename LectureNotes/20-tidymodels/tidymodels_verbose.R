# =========================================================
# BOSTON HOUSING ANALYSIS - MANUAL IMPLEMENTATION
# =========================================================
# This script demonstrates the step-by-step process of data preprocessing,
# model fitting, and evaluation - showing what tidymodels does "under the hood"
# =========================================================

# Basic libraries
library(tidyverse)

# Set seed for reproducibility
set.seed(123456)

# =========================================================
# 1. DATA LOADING AND EXPLORATION
# =========================================================
# Load the Boston Housing dataset
housing <- read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header = FALSE)
names(housing) <- c("crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", 
                    "rad", "tax", "ptratio", "b", "lstat", "medv")

# Quick exploration
head(housing)
summary(housing)

# =========================================================
# 2. MANUAL DATA SPLITTING
# =========================================================
# In tidymodels, initial_split() is used, but we can do this manually:

# Calculate number of observations for training (80%)
n <- nrow(housing)
train_size <- floor(0.8 * n)
train_indices <- sample(1:n, train_size)

# Create train and test sets
housing_train <- housing[train_indices, ]
housing_test <- housing[-train_indices, ]

cat("Training set size:", nrow(housing_train), "\n")
cat("Testing set size:", nrow(housing_test), "\n")

# =========================================================
# 3. MANUAL DATA PREPROCESSING
# =========================================================
# In tidymodels, recipe() and various step_* functions are used
# We'll do these transformations manually:

# 3.1 Log transform the outcome variable
housing_train$log_medv <- log(housing_train$medv)
housing_test$log_medv <- log(housing_test$medv)

# 3.2 Convert chas to factor
housing_train$chas <- as.factor(housing_train$chas)
housing_test$chas <- as.factor(housing_test$chas)

# 3.3 Create interaction term between crime and nox
housing_train$crim_nox <- housing_train$crim * housing_train$nox
housing_test$crim_nox <- housing_test$crim * housing_test$nox

# 3.4 Create polynomial terms for dis and nox
# For dis squared
housing_train$dis_sq <- housing_train$dis^2
housing_test$dis_sq <- housing_test$dis^2

# For nox squared
housing_train$nox_sq <- housing_train$nox^2
housing_test$nox_sq <- housing_test$nox^2

# =========================================================
# 4. MANUAL MODEL FITTING (OLS)
# =========================================================
# In tidymodels, the workflow() and fit() abstractions are used
# Let's do this manually with base R:

# Create formula for the model
ols_formula <- log_medv ~ crim + zn + indus + chas + nox + rm + age + 
    dis + rad + tax + ptratio + b + lstat + 
    crim_nox + dis_sq + nox_sq

# Fit the model
ols_model <- lm(ols_formula, data = housing_train)

# Examine results
summary(ols_model)

# =========================================================
# 5. MANUAL MODEL EVALUATION
# =========================================================
# In tidymodels, metrics functions like rmse() and rsq_trad() are used
# Let's calculate these metrics manually:

# 5.1 Make predictions on train and test sets
train_predictions <- predict(ols_model, newdata = housing_train)
test_predictions <- predict(ols_model, newdata = housing_test)

# 5.2 Calculate RMSE manually
# RMSE = sqrt(mean((actual - predicted)^2))
calc_rmse <- function(actual, predicted) {
    sqrt(mean((actual - predicted)^2))
}

train_rmse <- calc_rmse(housing_train$log_medv, train_predictions)
test_rmse <- calc_rmse(housing_test$log_medv, test_predictions)

cat("Training RMSE:", round(train_rmse, 3), "\n")
cat("Testing RMSE:", round(test_rmse, 3), "\n")

# 5.3 Calculate R-squared manually
# R² = 1 - (sum of squared residuals / total sum of squares)
calc_r_squared <- function(actual, predicted) {
    ss_residual <- sum((actual - predicted)^2)
    ss_total <- sum((actual - mean(actual))^2)
    1 - (ss_residual / ss_total)
}

train_r2 <- calc_r_squared(housing_train$log_medv, train_predictions)
test_r2 <- calc_r_squared(housing_test$log_medv, test_predictions)

cat("Training R²:", round(train_r2, 3), "\n")
cat("Testing R²:", round(test_r2, 3), "\n")

# =========================================================
# 6. MANUAL LASSO IMPLEMENTATION
# =========================================================
# In tidymodels, linear_reg() with engine="glmnet" is used
# Let's demonstrate a manual approach with glmnet:

library(glmnet)

# 6.1 Prepare matrix format required by glmnet
# First, create model matrix (handling factors appropriately)
x_train <- model.matrix(log_medv ~ crim + zn + indus + chas + nox + rm + age + 
                            dis + rad + tax + ptratio + b + lstat + 
                            crim_nox + dis_sq + nox_sq - 1, 
                        data = housing_train)

y_train <- housing_train$log_medv

x_test <- model.matrix(log_medv ~ crim + zn + indus + chas + nox + rm + age + 
                           dis + rad + tax + ptratio + b + lstat + 
                           crim_nox + dis_sq + nox_sq - 1, 
                       data = housing_test)

y_test <- housing_test$log_medv

# 6.2 Fit LASSO with fixed penalty (equivalent to penalty=0.5, mixture=1)
lambda <- 0.5
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = lambda)

# 6.3 Make predictions
lasso_train_pred <- predict(lasso_model, newx = x_train, s = lambda)
lasso_test_pred <- predict(lasso_model, newx = x_test, s = lambda)

# 6.4 Calculate metrics
lasso_train_rmse <- calc_rmse(y_train, lasso_train_pred)
lasso_test_rmse <- calc_rmse(y_test, lasso_test_pred)

lasso_train_r2 <- calc_r_squared(y_train, lasso_train_pred)
lasso_test_r2 <- calc_r_squared(y_test, lasso_test_pred)

cat("LASSO Training RMSE:", round(lasso_train_rmse, 3), "\n")
cat("LASSO Testing RMSE:", round(lasso_test_rmse, 3), "\n")
cat("LASSO Training R²:", round(lasso_train_r2, 3), "\n")
cat("LASSO Testing R²:", round(lasso_test_r2, 3), "\n")

# =========================================================
# 7. MANUAL CROSS-VALIDATION FOR LAMBDA SELECTION
# =========================================================
# In tidymodels, tune_grid() and vfold_cv() are used
# Let's do manual k-fold cross-validation:

# 7.1 Create a sequence of lambda values to try
lambda_seq <- 10^seq(-3, 1, length.out = 50)

# 7.2 Set up K-fold cross-validation
k <- 10
folds <- sample(1:k, nrow(housing_train), replace = TRUE)

# 7.3 Empty matrices to store CV results
cv_rmse <- matrix(NA, nrow = k, ncol = length(lambda_seq))
cv_r2 <- matrix(NA, nrow = k, ncol = length(lambda_seq))

# 7.4 Perform k-fold cross-validation for each lambda
for (i in 1:k) {
    # Split data
    cv_train_x <- x_train[folds != i, ]
    cv_train_y <- y_train[folds != i]
    cv_test_x <- x_train[folds == i, ]
    cv_test_y <- y_train[folds == i]
    
    # Fit model for all lambdas at once
    cv_model <- glmnet(cv_train_x, cv_train_y, alpha = 1, lambda = lambda_seq)
    
    # Make predictions for all lambdas
    cv_preds <- predict(cv_model, newx = cv_test_x, s = lambda_seq)
    
    # Calculate metrics for each lambda
    for (j in 1:length(lambda_seq)) {
        cv_rmse[i, j] <- calc_rmse(cv_test_y, cv_preds[, j])
        cv_r2[i, j] <- calc_r_squared(cv_test_y, cv_preds[, j])
    }
}

# 7.5 Calculate average metrics across folds
mean_cv_rmse <- colMeans(cv_rmse)
mean_cv_r2 <- colMeans(cv_r2)

# 7.6 Find best lambda value
best_lambda_index <- which.min(mean_cv_rmse)
best_lambda <- lambda_seq[best_lambda_index]

cat("Best lambda from CV:", best_lambda, "\n")
cat("Best CV RMSE:", round(mean_cv_rmse[best_lambda_index], 3), "\n")

# 7.7 Fit final model with best lambda
final_lasso <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)

# 7.8 Make predictions with best model
final_train_pred <- predict(final_lasso, newx = x_train, s = best_lambda)
final_test_pred <- predict(final_lasso, newx = x_test, s = best_lambda)

# 7.9 Calculate final metrics
final_train_rmse <- calc_rmse(y_train, final_train_pred)
final_test_rmse <- calc_rmse(y_test, final_test_pred)
final_train_r2 <- calc_r_squared(y_train, final_train_pred)
final_test_r2 <- calc_r_squared(y_test, final_test_pred)

cat("Final LASSO with best lambda:\n")
cat("Training RMSE:", round(final_train_rmse, 3), "\n")
cat("Testing RMSE:", round(final_test_rmse, 3), "\n")
cat("Training R²:", round(final_train_r2, 3), "\n")
cat("Testing R²:", round(final_test_r2, 3), "\n")

# 8. BONUS: EXAMINE COEFFICIENTS
# ========================================================
# Look at which variables were selected by LASSO
lasso_coef <- coef(final_lasso, s = best_lambda)
lasso_coef

# Compare to OLS coefficients
ols_coef <- coef(ols_model)
ols_coef