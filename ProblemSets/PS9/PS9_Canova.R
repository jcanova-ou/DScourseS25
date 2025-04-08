install.packages("tidymodels")
install.packages("glmnet")
install.packages("mlbench")


library(tidymodels)
library(glmnet)
library(mlbench)

#4. Load the housing data from UCI, following the example in the lecture notes (Lecture 20).
{
  # Load the Boston Housing dataset
  data("BostonHousing")

  # View the structure
  str(BostonHousing)

  # Get a summary
  summary(BostonHousing)
}

#5. Set the seed to 123456.
{
  set.seed(123456)
}

#6. Create two data sets called housing_train and housing_test using the initial_split()
#function from the rsample package, following the example in the lecture notes.

{
  # Split the data into training and testing sets
  housing_split <- initial_split(BostonHousing, prop = 0.8, strata = "medv")
  
  # Create training and testing datasets
  housing_train <- training(housing_split)
  housing_test <- testing(housing_split)
  
  # Check the dimensions of the datasets
  dim(housing_train)
  dim(housing_test)

}


#7. Create a new recipe() that takes the log of the housing value, converts chas to a factor, 
{
# creates 6th degree polynomials of each of the continuous features (i.e. everything except chas), 
# and linear interactions of each. To do so, add the following code to your script. 

# What is the dimension of your training data (housing_train)? 14
# How many more X variables do you have than in the original housing data? 14

    housing_recipe <- recipe (medv ~ . , data = BostonHousing ) %>%
# convert outcome variable to logs
    step_log (all_outcomes()) %>%
# convert 0/1 chas to a factor
#    step_bin2factor(chas) %>%
# create interaction term between crime and nox
    step_interact(terms = ~ crim : zn : indus :rm: age : rad : tax : 
                    ptratio : b : lstat : dis : nox ) %>%
# create square terms of some continuous variables
    step_poly ( crim , zn , indus ,rm , age , rad , tax , ptratio ,b ,
             lstat , dis , nox , degree =6)
# Run the recipe
    housing_prep <- housing_recipe %>% prep (housing_train , retain  = TRUE)
    housing_train_prepped <- housing_prep %>% juice
    housing_test_prepped <- housing_prep %>% bake (new_data = housing_test )
    
# create x and y training and test data
    housing_train_x <- housing_train_prepped %>% select ( - medv )
    housing_test_x <- housing_test_prepped %>% select ( - medv )
    housing_train_y <- housing_train_prepped %>% select ( medv )
    housing_test_y <- housing_test_prepped %>% select ( medv )

}



#8. Following the example from the lecture notes, estimate a LASSO model to predict
#log median house value, where the penalty parameter λ is tuned by 6-fold cross validation. 

#What is the optimal value of λ? 0.0233
#What is the in-sample RMSE? 0.187
#What is the out-of-sample RMSE (i.e. the RMSE in the test data)? 0.149
{
  # Create a LASSO model specification
  lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
    set_engine("glmnet") %>%
    set_mode("regression")
  
  # Create a workflow
  lasso_workflow <- workflow() %>%
    add_recipe(housing_recipe) %>%
    add_model(lasso_spec)
  
  # Set up a grid for tuning
  lasso_grid <- grid_regular(penalty(), levels = 10)
  
  # Perform cross-validation
  # Start with a fresh recipe
  housing_recipe <- recipe(medv ~ ., data = BostonHousing) %>%
    # Log transform the outcome
    step_log(all_outcomes()) %>%
    # IMPORTANT: Convert all factor variables to dummy variables
    step_dummy(all_nominal_predictors()) %>%
    # Now proceed with your interactions (which will now only involve numeric columns)
    step_interact(terms = ~ crim:zn:indus:rm:age:rad:tax:ptratio:b:lstat:dis:nox) %>%
    # Create polynomial terms
    step_poly(crim, zn, indus, rm, age, rad, tax, ptratio, b, lstat, dis, nox, degree = 6)
  
  # Prepare the recipe
  housing_prep <- housing_recipe %>% prep(housing_train, retain = TRUE)
  housing_train_prepped <- housing_prep %>% juice()
  housing_test_prepped <- housing_prep %>% bake(new_data = housing_test)
  
  # Create x and y training and test data
  housing_train_x <- housing_train_prepped %>% select(-medv)
  housing_test_x <- housing_test_prepped %>% select(-medv)
  housing_train_y <- housing_train_prepped %>% select(medv)
  housing_test_y <- housing_test_prepped %>% select(medv)
  
  # Define the LASSO model
  lasso_model <- linear_reg(penalty = tune(), mixture = 1) %>%
    set_engine("glmnet")
  
  # Create the workflow
  lasso_workflow <- workflow() %>%
    add_model(lasso_model) %>%
    add_recipe(housing_recipe)
  
  # Define penalty grid
  lasso_grid <- grid_regular(penalty(), levels = 50)
  
  # Perform cross-validation
  lasso_res <- tune_grid(
    lasso_workflow,
    resamples = vfold_cv(housing_train, v = 6),
    grid = lasso_grid,
    metrics = metric_set(rmse)
  )
  
  
  
  
 # Define the LASSO model that handles factors
lasso_model <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet", importance = "impurity")

# Create the workflow with your recipe
lasso_workflow <- workflow() %>%
  add_recipe(housing_recipe) %>%
  add_model(lasso_model)

# Define your penalty grid (assuming this is how you defined lasso_grid)
lasso_grid <- grid_regular(penalty(), levels = 50)

# Perform cross-validation with the same parameters you were using
lasso_res <- tune_grid(
  lasso_workflow,
  resamples = vfold_cv(housing_train, v = 6),
  grid = lasso_grid,
  metrics = metric_set(rmse)
)
  
  
  
  
  
  
  
   lasso_model <- linear_reg(penalty = tune(), mixture = 1) %>%
    set_engine("glmnet", importance = "impurity")
  
  lasso_workflow <- workflow() %>%
    add_recipe(housing_recipe) %>%
    add_model(lasso_model)
  
  
  
  
  
  
  lasso_res <- tune_grid(
    lasso_workflow,
    resamples = vfold_cv(housing_train, v = 6),
    grid = lasso_grid,
    metrics = metric_set(rmse)
  )
  
  # Get the best penalty value
  best_lasso <- select_best(lasso_res, metric = "rmse")
  print(best_lasso)
  
  # Finalize the workflow with the best penalty
  final_lasso_workflow <- finalize_workflow(lasso_workflow, best_lasso)
  
  # Fit the final model on the training data
  final_lasso_fit <- fit(final_lasso_workflow, data = housing_train)
  
  # Get RMSE on training data
  # Extract fitted model from workflow
  lasso_model_only <- extract_fit_parsnip(final_lasso_fit)
  
  # Create predictions using just the model (bypassing the recipe)
  train_rmse <- predict(lasso_model_only, new_data = housing_train_x) %>% 
    bind_cols(housing_train_y) %>% 
    rmse(truth = medv, estimate = .pred)
  
  # Get RMSE on test data
  # Extract fitted model from workflow
  lasso_model_only <- extract_fit_parsnip(final_lasso_fit)
  
  # Create predictions using just the model
  test_rmse <- predict(lasso_model_only, new_data = housing_test_x) %>% 
    bind_cols(housing_test_y) %>% 
    rmse(truth = medv, estimate = .pred)
  

}


#  9. Repeat the previous question, but now estimate a ridge regression model where again
#the penalty parameter λ is tuned by 6-fold CV. 
# What is the optimal value of λ now? 0.0000000001
# What is the out-of-sample RMSE (i.e. the RMSE in the test data)? 0.157

{
# Start with the same recipe as before
housing_recipe <- recipe(medv ~ ., data = BostonHousing) %>%
  # Log transform the outcome
  step_log(all_outcomes()) %>%
  # Convert all factor variables to dummy variables
  step_dummy(all_nominal_predictors()) %>%
  # Proceed with your interactions
  step_interact(terms = ~ crim:zn:indus:rm:age:rad:tax:ptratio:b:lstat:dis:nox) %>%
  # Create polynomial terms
  step_poly(crim, zn, indus, rm, age, rad, tax, ptratio, b, lstat, dis, nox, degree = 6)

# Prepare the recipe
housing_prep <- housing_recipe %>% prep(housing_train, retain = TRUE)
housing_train_prepped <- housing_prep %>% juice()
housing_test_prepped <- housing_prep %>% bake(new_data = housing_test)

# Create x and y training and test data
housing_train_x <- housing_train_prepped %>% select(-medv)
housing_test_x <- housing_test_prepped %>% select(-medv)
housing_train_y <- housing_train_prepped %>% select(medv)
housing_test_y <- housing_test_prepped %>% select(medv)

# Define the Ridge regression model (mixture = 0 instead of 1)
ridge_model <- linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet")

# Create the workflow
ridge_workflow <- workflow() %>%
  add_recipe(housing_recipe) %>%
  add_model(ridge_model)

# Define penalty grid
ridge_grid <- grid_regular(penalty(), levels = 50)

# Perform 6-fold cross-validation
ridge_res <- tune_grid(
  ridge_workflow,
  resamples = vfold_cv(housing_train, v = 6),
  grid = ridge_grid,
  metrics = metric_set(rmse)
)

# Get the best penalty value (lambda)
best_ridge <- select_best(ridge_res, metric = "rmse")
print("Optimal value of lambda for Ridge regression:")
print(best_ridge)

# Finalize the workflow with the best penalty
final_ridge_workflow <- finalize_workflow(ridge_workflow, best_ridge)

# Fit the final model on the training data
final_ridge_fit <- fit(final_ridge_workflow, data = housing_train)

# Extract fitted model from workflow
ridge_model_only <- extract_fit_parsnip(final_ridge_fit)

# Calculate training RMSE
train_ridge_rmse <- predict(ridge_model_only, new_data = housing_train_x) %>% 
  bind_cols(housing_train_y) %>% 
  rmse(truth = medv, estimate = .pred)
print("Training RMSE:")
print(train_ridge_rmse)

# Calculate test RMSE (out-of-sample)
test_ridge_rmse <- predict(ridge_model_only, new_data = housing_test_x) %>% 
  bind_cols(housing_test_y) %>% 
  rmse(truth = medv, estimate = .pred)
print("Out-of-sample RMSE (test data):")
print(test_ridge_rmse)

# Optional: Visualize the tuning results
autoplot(ridge_res) +
  labs(title = "Ridge Regression Tuning Results",
       subtitle = "RMSE vs Penalty Parameter")
}

#  10. In your .tex file, answer the questions posed in the preceding questions. 
#Would you be able to estimate a simple linear regression model on a data set that had more columns than rows? 
#Using the RMSE values of each of the tuned models in the previous two #questions, comment on where your model stands in terms of the bias-variance tradeoff.

