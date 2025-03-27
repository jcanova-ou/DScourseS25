library(mice)
library(modelsummary)
library(tidyverse)

#Question 7
# Remove missing values for hgc and tenure
wage_cleaned <- wages[complete.cases(wages[, c("hgc", "tenure")]), ]
datasummary_skim(wage_cleaned, output = "latex")

#Question 8
# Investigate missingness

# base model for comparison
model <- lm(logwage ~ hgc + tenure + age + married + college, data = wage_cleaned)
modelsummary(model)

#part 1 only completed cases
wagep1 <- wage_cleaned[complete.cases(wage_cleaned$logwage), ]
model1 <- lm(logwage ~ hgc + tenure + age + married + college, data = wagep1)
modelsummary(model1)  

#part 2 mean impute logwage
# Calculate the mean of log wages
mean_logwage <- mean(wage_cleaned$logwage, na.rm = TRUE)

# Create a new data frame with mean imputation
wagep2 <- wage_cleaned %>%
  mutate(logwage = ifelse(is.na(logwage), mean_logwage, logwage))

# Estimate the regression with imputed data
modelp2 <- lm(logwage ~ hgc + tenure + age + married + college, data = wagep2)

modelsummary(modelp2)


#part 3 

#Predict log wages for observations with missing values
# Create a copy of the original dataset
wagep3 <- wagep1

# Find indices of missing log wages
missing_indices <- which(is.na(wagep1$logwage))

# Predict log wages for missing observations using the complete cases model
wagep3$logwage[missing_indices] <- predict(model1, newdata = wagep1[missing_indices, ])

#Run the full regression on the dataset with predicted values
model3 <- lm(logwage ~ hgc + tenure + age + married + college, data = wagep3)

modelsummary(model3)


#Part 4

# Prepare the data for imputation
# Select variables for imputation
imputation_vars <- c("logwage", "hgc", "tenure", "age", "married", "college")
wagep1_imp_data <- wages[, imputation_vars]

# Inspect missing data patterns
md.pattern(wagep1_imp_data)

# Set seed for reproducibility
set.seed(123)

# Perform multiple imputation
# Use predictive mean matching for continuous variables
mice_imputation <- mice(wagep1_imp_data, 
                        m = 5,  # 5 imputed datasets
                        method = "pmm")  # Predictive mean matching

# Fit regression model on each imputed dataset
models <- with(mice_imputation, 
               lm(logwage ~ hgc + tenure + age + married + college))

# Pool the results across multiple imputations
pooled_results <- pool(models)

# Summary of pooled results
summary(pooled_results)

modelsummary(models$analyses, output = "latex")


