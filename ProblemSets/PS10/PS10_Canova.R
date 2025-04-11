# Load required libraries
library(tidymodels)
library(tidyverse)
library(magrittr)
library(glmnet)
library(rpart)
library(nnet)
library(kknn)
library(kernlab)

# Set seed for reproducibility
set.seed(100)

# Load the data from URL
income <- read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", col_names = FALSE)
names(income) <- c("age","workclass","fnlwgt","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours","native.country","high.earner")

# Clean up the data as in the original code
# Drop unnecessary columns
income %<>% select(-native.country, -fnlwgt, -education.num)

# Make sure continuous variables are formatted as numeric
income %<>% mutate(across(c(age, hours, capital.gain, capital.loss), as.numeric))

# Make sure discrete variables are formatted as factors
income %<>% mutate(across(c(high.earner, education, marital.status, race, workclass, occupation, relationship, sex), as.factor))

# Combine levels of factor variables that currently have too many levels
income %<>% mutate(education = fct_collapse(education,
                                            Advanced    = c("Masters","Doctorate","Prof-school"), 
                                            Bachelors   = c("Bachelors"), 
                                            SomeCollege = c("Some-college","Assoc-acdm","Assoc-voc"),
                                            HSgrad      = c("HS-grad","12th"),
                                            HSdrop      = c("11th","9th","7th-8th","1st-4th","10th","5th-6th","Preschool") 
),
marital.status = fct_collapse(marital.status,
                              Married      = c("Married-civ-spouse","Married-spouse-absent","Married-AF-spouse"), 
                              Divorced     = c("Divorced","Separated"), 
                              Widowed      = c("Widowed"), 
                              NeverMarried = c("Never-married")
), 
race = fct_collapse(race,
                    White = c("White"), 
                    Black = c("Black"), 
                    Asian = c("Asian-Pac-Islander"), 
                    Other = c("Other","Amer-Indian-Eskimo")
), 
workclass = fct_collapse(workclass,
                         Private = c("Private"), 
                         SelfEmp = c("Self-emp-not-inc","Self-emp-inc"), 
                         Gov     = c("Federal-gov","Local-gov","State-gov"), 
                         Other   = c("Without-pay","Never-worked","?")
), 
occupation = fct_collapse(occupation,
                          BlueCollar  = c("?","Craft-repair","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Transport-moving"), 
                          WhiteCollar = c("Adm-clerical","Exec-managerial","Prof-specialty","Sales","Tech-support"), 
                          Services    = c("Armed-Forces","Other-service","Priv-house-serv","Protective-serv")
)
)

# Split the data into training and testing sets
income_split <- initial_split(income, prop = 0.8)
income_train <- training(income_split)
income_test <- testing(income_split)

# Create 3-fold cross-validation object
cv_folds <- vfold_cv(income_train, v = 3)

# Create a common formula for all models
model_formula <- high.earner ~ age + workclass + education + marital.status + 
  occupation + relationship + race + sex + capital.gain + 
  capital.loss + hours

# Create metrics collection
eval_metrics <- metric_set(accuracy, precision, recall)

# Print data summary for verification
print("Data structure check:")
print(str(income_train))
print(table(income_train$high.earner))

# ------------------------------
# MODEL 1: LOGISTIC REGRESSION
# ------------------------------
print('Starting LOGISTIC REGRESSION')
# Set up logistic regression with tuning
tune_log_reg_spec <- logistic_reg(
  penalty = tune(),
  mixture = 1  # Use LASSO (mixture=1)
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# Define parameter grid for logistic regression
log_reg_grid <- grid_regular(
  penalty(range = c(-3, 0), trans = log10_trans()),
  levels = 50
)

# Create workflow
log_reg_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(tune_log_reg_spec)

# Tune logistic regression
log_reg_results <- log_reg_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid = log_reg_grid,
    metrics = eval_metrics
  )

# Get best parameters
best_log_reg_params <- log_reg_results %>% select_best(metric = "accuracy")
print("Best logistic regression parameters:")
print(best_log_reg_params)

# ------------------------------
# MODEL 2: DECISION TREE
# ------------------------------
print('Starting TREE')
# Set up decision tree with tuning
tune_tree_spec <- decision_tree(
  min_n = tune(), 
  tree_depth = tune(), 
  cost_complexity = tune()
) %>% 
  set_engine("rpart") %>%
  set_mode("classification")

# Define parameter grid for tree
tree_grid <- expand_grid(
  cost_complexity = seq(.001, .2, by=.05),
  min_n = seq(10, 100, by=10),
  tree_depth = seq(5, 20, by=5)
)

# Create workflow
tree_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(tune_tree_spec)

# Tune tree model
tree_results <- tree_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid = tree_grid,
    metrics = eval_metrics
  )

# Get best parameters
best_tree_params <- tree_results %>% select_best(metric = "accuracy")
print("Best tree parameters:")
print(best_tree_params)

# ------------------------------
# MODEL 3: NEURAL NETWORK
# ------------------------------
print('Starting NEURAL NETWORK')
# Set up neural network with tuning
tune_nnet_spec <- mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = 100
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

# Define parameter grid for neural network
nnet_grid <- expand_grid(
  hidden_units = seq(1, 10),
  penalty = 10^seq(-5, -1, length.out = 10)
)

# Create workflow
nnet_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(tune_nnet_spec)

# Tune neural network
nnet_results <- nnet_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid = nnet_grid,
    metrics = eval_metrics
  )

# Get best parameters
best_nnet_params <- nnet_results %>% select_best(metric = "accuracy")
print("Best neural network parameters:")
print(best_nnet_params)

# ------------------------------
# MODEL 4: K-NEAREST NEIGHBORS
# ------------------------------
print('Starting KNN')
# Set up KNN with tuning
tune_knn_spec <- nearest_neighbor(
  neighbors = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Define parameter grid for KNN
knn_grid <- tibble(neighbors = seq(1, 30))

# Create workflow
knn_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(tune_knn_spec)

# Tune KNN
knn_results <- knn_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid = knn_grid,
    metrics = eval_metrics
  )

# Get best parameters
best_knn_params <- knn_results %>% select_best(metric = "accuracy")
print("Best KNN parameters:")
print(best_knn_params)

# ------------------------------
# MODEL 5: SUPPORT VECTOR MACHINE
# ------------------------------
print('Starting SVM')
# Set up SVM with tuning
tune_svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Define parameter grid for SVM
svm_grid <- expand_grid(
  cost = c(2^(-2), 2^(-1), 2^0, 2^1, 2^2, 2^10),
  rbf_sigma = c(2^(-2), 2^(-1), 2^0, 2^1, 2^2, 2^10)
)

# Create workflow
svm_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(tune_svm_spec)

# Tune SVM
svm_results <- svm_wf %>%
  tune_grid(
    resamples = cv_folds,
    grid = svm_grid,
    metrics = eval_metrics
  )

# Get best parameters
best_svm_params <- svm_results %>% select_best(metric = "accuracy")
print("Best SVM parameters:")
print(best_svm_params)

# ------------------------------
# CREATE FINAL MODELS WITH BEST PARAMETERS
# ------------------------------

# Create final model specifications with explicit parameters
final_log_reg_spec <- logistic_reg(
  penalty = best_log_reg_params$penalty,
  mixture = 1
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

final_tree_spec <- decision_tree(
  min_n = best_tree_params$min_n,
  tree_depth = best_tree_params$tree_depth,
  cost_complexity = best_tree_params$cost_complexity
) %>% 
  set_engine("rpart") %>%
  set_mode("classification")

final_nnet_spec <- mlp(
  hidden_units = best_nnet_params$hidden_units,
  penalty = best_nnet_params$penalty,
  epochs = 100
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

final_knn_spec <- nearest_neighbor(
  neighbors = best_knn_params$neighbors
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

final_svm_spec <- svm_rbf(
  cost = best_svm_params$cost,
  rbf_sigma = best_svm_params$rbf_sigma
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Create final workflows with best parameters
final_log_reg_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(final_log_reg_spec)

final_tree_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(final_tree_spec)

final_nnet_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(final_nnet_spec)

final_knn_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(final_knn_spec)

final_svm_wf <- workflow() %>%
  add_formula(model_formula) %>%
  add_model(final_svm_spec)

# ------------------------------
# EVALUATE ALL FINAL MODELS
# ------------------------------

# Function to train and evaluate a model safely
evaluate_model_safely <- function(workflow_obj, model_name) {
  print(paste("Evaluating", model_name))
  
  # Try to fit the model
  model_fit <- tryCatch({
    workflow_obj %>% fit(income_train)
  }, error = function(e) {
    print(paste("Error fitting", model_name, ":", e$message))
    return(NULL)
  })
  
  # If model fitting failed, return NULL
  if(is.null(model_fit)) {
    return(NULL)
  }
  
  # Try to predict classes
  predictions <- tryCatch({
    predict(model_fit, income_test) %>%
      bind_cols(income_test %>% select(high.earner))
  }, error = function(e) {
    print(paste("Error predicting with", model_name, ":", e$message))
    return(NULL)
  })
  
  # If predictions failed, return NULL
  if(is.null(predictions)) {
    return(NULL)
  }
  
  # Calculate metrics
  metrics_result <- tryCatch({
    predictions %>% metrics(truth = high.earner, estimate = .pred_class)
  }, error = function(e) {
    print(paste("Error calculating metrics for", model_name, ":", e$message))
    return(NULL)
  })
  
  # Try to get probability predictions if possible
  prob_predictions <- tryCatch({
    predict(model_fit, income_test, type = "prob") %>%
      bind_cols(predictions)
  }, error = function(e) {
    print(paste("Warning: Probability predictions not available for", model_name))
    return(predictions)  # Return class predictions only
  })
  
  return(list(
    model_name = model_name,
    metrics = metrics_result,
    predictions = prob_predictions
  ))
}

# Evaluate all models one by one
log_reg_eval <- evaluate_model_safely(final_log_reg_wf, "Logistic Regression")
tree_eval <- evaluate_model_safely(final_tree_wf, "Decision Tree")
nnet_eval <- evaluate_model_safely(final_nnet_wf, "Neural Network")
knn_eval <- evaluate_model_safely(final_knn_wf, "K-Nearest Neighbors")
svm_eval <- evaluate_model_safely(final_svm_wf, "Support Vector Machine")

# Collect all successful evaluations
all_evaluations <- list()
if(!is.null(log_reg_eval)) all_evaluations[["Logistic Regression"]] <- log_reg_eval
if(!is.null(tree_eval)) all_evaluations[["Decision Tree"]] <- tree_eval
if(!is.null(nnet_eval)) all_evaluations[["Neural Network"]] <- nnet_eval
if(!is.null(knn_eval)) all_evaluations[["K-Nearest Neighbors"]] <- knn_eval
if(!is.null(svm_eval)) all_evaluations[["Support Vector Machine"]] <- svm_eval

# Create summary table of results
if(length(all_evaluations) > 0) {
  results_table <- map_dfr(all_evaluations, function(eval_result) {
    eval_result$metrics %>%
      filter(.metric == "accuracy") %>%
      select(.estimate) %>%
      mutate(Model = eval_result$model_name)
  })
  
  # Display results table
  print("Model Accuracy Comparison:")
  print(results_table %>% 
          arrange(desc(.estimate)) %>%
          rename(Accuracy = .estimate))
  
  # Identify best model
  best_model <- results_table %>%
    arrange(desc(.estimate)) %>%
    slice(1) %>%
    pull(Model)
  
  print(paste("Best performing model:", best_model))
} else {
  print("No models were successfully evaluated.")
}

# Create a table of best parameters for each model
tuning_params <- tibble(
  Model = c("Logistic Regression", "Decision Tree", "Neural Network", "K-Nearest Neighbors", "Support Vector Machine"),
  Parameters = c(
    paste("penalty =", round(best_log_reg_params$penalty, 5)),
    paste("min_n =", best_tree_params$min_n, ", depth =", best_tree_params$tree_depth, 
          ", complexity =", round(best_tree_params$cost_complexity, 5)),
    paste("hidden_units =", best_nnet_params$hidden_units, ", penalty =", round(best_nnet_params$penalty, 5)),
    paste("neighbors =", best_knn_params$neighbors),
    paste("cost =", best_svm_params$cost, ", sigma =", best_svm_params$rbf_sigma)
  )
)

print("Optimal tuning parameters:")
print(tuning_params)
