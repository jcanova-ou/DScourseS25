install.packages("nloptr")
library(nloptr)
library(modelsummary)


#---------------------------
#        Question 4
#---------------------------

set.seed(100)

# Set dimensions
N <- 100000
K <- 10

# Create the matrix with normally distributed random numbers
X <- matrix(rnorm((K-1)*N), nrow=N, ncol=K-1)

# Add a column of 1's as the first column
X <- cbind(rep(1, N), X)


# Set the standard deviation
sigma <- 0.5

# Create epsilon vector of length N with N(0, σ²) distribution
eps <- rnorm(N, mean = 0, sd = sigma)

# Verify length and distribution
length(eps)
mean(eps)
var(eps)  # This should be approximately 0.25

# Create beta vector with the given values
beta <- c(1.5, -1, -0.25, 0.75, 3.5, -2, 0.5, 1, 1.25, 2)

# Verify length and values
length(beta)
beta

# Generate Y = X*beta + eps
Y <- X %*% beta + eps

# Check dimensions
length(Y)
head(Y)  # View first few elements

#---------------------------------
#           Question 5
#---------------------------------
# Compute OLS estimate using the closed-form solution
# Beta_hat_OLS = (X'X)^(-1)X'Y

# Step 1: Calculate X'X
XtX <- t(X) %*% X

# Step 2: Calculate (X'X)^(-1)
XtX_inv <- solve(XtX)

# Step 3: Calculate X'Y
XtY <- t(X) %*% Y

# Step 4: Calculate βˆOLS = (X'X)^(-1)X'Y
beta_OLS <- XtX_inv %*% XtY

# Print the OLS estimate
beta_OLS

# Compare with true beta
comparison <- data.frame(
  True_Beta = beta,
  Estimated_Beta = as.vector(beta_OLS),
  Difference = as.vector(beta_OLS) - beta
)
print(comparison)

#---------------------------------
#           Question 6
#---------------------------------

# Function to compute the sum of squared residuals (SSR)
compute_ssr <- function(X, Y, beta) {
  residuals <- Y - X %*% beta
  sum(residuals^2)
}

# Function to compute the gradient of SSR with respect to beta
compute_gradient <- function(X, Y, beta) {
  residuals <- Y - X %*% beta
  -2 * t(X) %*% residuals
}

# Gradient descent implementation
gradient_descent_ols <- function(X, Y, learning_rate = 0.0000003, max_iterations = 10000, tolerance = 1e-6) {
  # Initialize beta to zeros
  beta_gd <- matrix(0, nrow = ncol(X), ncol = 1)
  
  # Store SSR history to monitor convergence
  ssr_history <- numeric(max_iterations)
  
  # Initial SSR
  ssr_old <- compute_ssr(X, Y, beta_gd)
  
  # Gradient descent iterations
  for (i in 1:max_iterations) {
    # Compute gradient
    gradient <- compute_gradient(X, Y, beta_gd)
    
    # Update beta
    beta_gd <- beta_gd - learning_rate * gradient
    
    # Compute new SSR
    ssr_new <- compute_ssr(X, Y, beta_gd)
    ssr_history[i] <- ssr_new
    
    # Check for convergence
    if (abs(ssr_new - ssr_old) < tolerance) {
      cat("Converged after", i, "iterations\n")
      break
    }
    
    # Update old SSR
    ssr_old <- ssr_new
    
    # Print progress every 1000 iterations
    if (i %% 1000 == 0) {
      cat("Iteration", i, ": SSR =", ssr_new, "\n")
    }
  }
  
  # If reached max iterations without convergence
  if (i == max_iterations) {
    cat("Maximum iterations reached without convergence\n")
  }
  
  return(list(beta = beta_gd, ssr_history = ssr_history[1:i]))
}

# Run gradient descent with learning rate 0.0000003
gd_result <- gradient_descent_ols(X, Y, learning_rate = 0.0000003)

# Compare results
comparison <- data.frame(
  True_Beta = beta,
  Closed_Form_OLS = as.vector(beta_OLS),
  Gradient_Descent_OLS = as.vector(gd_result$beta),
  GD_vs_True_Difference = as.vector(gd_result$beta) - beta
)
print(comparison)

# Optional: Plot SSR history to visualize convergence
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  ggplot(data.frame(Iteration = 1:length(gd_result$ssr_history), 
                    SSR = gd_result$ssr_history), 
         aes(x = Iteration, y = SSR)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Convergence of Gradient Descent", 
         y = "Sum of Squared Residuals")
}

#---------------------------------
#           Question 7
#---------------------------------

# Define the objective function (sum of squared residuals)
objective_fn <- function(beta, X, Y) {
  residuals <- Y - X %*% beta
  sum(residuals^2)
}

# Define the gradient function for the objective
gradient_fn <- function(beta, X, Y) {
  residuals <- Y - X %*% beta
  -2 * t(X) %*% residuals
}

# Starting values (initialize to zeros)
beta_init <- rep(0, ncol(X))

# L-BFGS optimization
lbfgs_result <- nloptr(
  x0 = beta_init,
  eval_f = objective_fn,
  eval_grad_f = gradient_fn,
  opts = list(
    algorithm = "NLOPT_LD_LBFGS",
    xtol_rel = 1e-8,
    maxeval = 1000,
    print_level = 1
  ),
  X = X,
  Y = Y
)

# Nelder-Mead optimization
nm_result <- nloptr(
  x0 = beta_init,
  eval_f = objective_fn,
  opts = list(
    algorithm = "NLOPT_LN_NELDERMEAD",
    xtol_rel = 1e-8,
    maxeval = 5000,
    print_level = 1
  ),
  X = X,
  Y = Y
)

# Extract the results
beta_lbfgs <- lbfgs_result$solution
beta_nm <- nm_result$solution

# Compare all methods
comparison <- data.frame(
  True_Beta = beta,
  Closed_Form_OLS = as.vector(beta_OLS),
  Gradient_Descent = as.vector(gd_result$beta),
  LBFGS = beta_lbfgs,
  Nelder_Mead = beta_nm
)

# Add comparison metrics
comparison$LBFGS_vs_OLS_Diff <- comparison$LBFGS - comparison$Closed_Form_OLS
comparison$NM_vs_OLS_Diff <- comparison$Nelder_Mead - comparison$Closed_Form_OLS

# Print comparison
print(comparison)

# Summary of algorithms
cat("\nSummary of optimization results:\n")
cat("L-BFGS objective value:", lbfgs_result$objective, "\n")
cat("Nelder-Mead objective value:", nm_result$objective, "\n")
cat("L-BFGS iterations:", lbfgs_result$iterations, "\n")
cat("Nelder-Mead iterations:", nm_result$iterations, "\n")

#---------------------------------
#           Question 8
#---------------------------------

gradient <- function ( theta ,Y , X ) {
  grad <- as.vector ( rep (0 , length ( theta ) ) )
  beta <- theta [1:( length ( theta ) -1) ]
  sig <- theta [ length ( theta ) ]
  grad [1:( length ( theta ) -1) ] <- -t( X ) %* %( Y - X %* % beta )/( sig ^2)
  grad [ length ( theta ) ] <- dim ( X ) [1]/ sig - crossprod (Y - X %* % beta )/( sig
                                                                                   ^3)
  return ( grad )
}
print(gradient)


#---------------------------------
#           Question 9
#---------------------------------

# Compute OLS estimate using lm() function
ols_model <- lm(Y ~ X - 1)
modelsummary(ols_model)
