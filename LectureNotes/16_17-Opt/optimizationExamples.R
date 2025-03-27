library(nloptr)

#----------------------------------
# OLS estimation with L-BFGS
#----------------------------------

## Our objective function
objfun <- function(beta,y,X) {
return (sum((y-X%*%beta)^2))
# equivalently, if we want to use matrix algebra:
# return ( crossprod(y-X%*%beta) )
}

## Gradient of our objective function
gradient <- function(beta,y,X) {
return ( as.vector(-2*t(X)%*%(y-X%*%beta)) )
}

## read in the data
y <- iris$Sepal.Length
X <- model.matrix(~Sepal.Width+Petal.Length+Petal.Width+Species,iris)

## initial values
beta0 <- runif(dim(X)[2]) #start at uniform random numbers equal to number of coefficients

## Algorithm parameters
options <- list("algorithm"="NLOPT_LD_LBFGS","xtol_rel"=1.0e-6,"maxeval"=1e3)

## Optimize!
result <- nloptr( x0=beta0,eval_f=objfun,eval_grad_f=gradient,opts=options,y=y,X=X)
print(result)

## Check solution
print(summary(lm(Sepal.Length~Sepal.Width+Petal.Length+Petal.Width+Species,data=iris)))


#----------------------------------
# MLE estimation with Nelder Mead
#----------------------------------
## Our objective function
objfun  <- function(theta,y,X) {
# need to slice our parameter vector into beta and sigma components
beta    <- theta[1:(length(theta)-1)]
sig     <- theta[length(theta)]
# write objective function as *negative* log likelihood (since NLOPT minimizes)
loglike <- -sum( -.5*(log(2*pi*(sig^2)) + ((y-X%*%beta)/sig)^2) ) 
return (loglike)
}

## read in the data
y <- iris$Sepal.Length
X <- model.matrix(~Sepal.Width+Petal.Length+Petal.Width+Species,iris)

## initial values
theta0 <- runif(dim(X)[2]+1) #start at uniform random numbers equal to number of coefficients
#theta0 <- append(as.vector(summary(lm(Sepal.Length~Sepal.Width+Petal.Length+Petal.Width+Species,data=iris))$coefficients[,1]),runif(1))

## Algorithm parameters
options <- list("algorithm"="NLOPT_LN_NELDERMEAD","xtol_rel"=1.0e-6,"maxeval"=1e4)

## Optimize!
result <- nloptr( x0=theta0,eval_f=objfun,opts=options,y=y,X=X)
print(result)
betahat  <- result$solution[1:(length(result$solution)-1)]
sigmahat <- result$solution[length(result$solution)]

## Check solution
print(summary(lm(Sepal.Length~Sepal.Width+Petal.Length+Petal.Width+Species,data=iris)))



#----------------------------------
# MLE estimation with L-BFGS
#----------------------------------
## Our objective function
objfun  <- function(theta,y,X) {
# need to slice our parameter vector into beta and sigma components
beta    <- theta[1:(length(theta)-1)]
sig     <- theta[length(theta)]
# write objective function as *negative* log likelihood (since NLOPT minimizes)
loglike <- -sum( -.5*(log(2*pi*(sig^2)) + ((y-X%*%beta)/sig)^2) ) 
return (loglike)
}

## Gradient of the objective function
gradient <- function (theta,y,X) {
grad     <- as.vector(rep(0,length(theta)))
beta     <- theta [1:(length(theta)-1)]
sig      <- theta [length(theta)]
grad[1:(length(theta)-1)] <- -t(X)%*%(y - X%*%beta)/(sig^2)
grad[length(theta)]       <- dim(X)[1]/sig-crossprod (y-X%*%beta)/(sig^3)
return ( grad )
}

## read in the data
y <- iris$Sepal.Length
X <- model.matrix(~Sepal.Width+Petal.Length+Petal.Width+Species,iris)

## initial values
theta0 <- runif(dim(X)[2]+1) #start at uniform random numbers equal to number of coefficients
theta0 <- append(as.vector(summary(lm(Sepal.Length~Sepal.Width+Petal.Length+Petal.Width+Species,data=iris))$coefficients[,1]),runif(1))

## Algorithm parameters
options <- list("algorithm"="NLOPT_LD_LBFGS","xtol_rel"=1.0e-6,"maxeval"=1e4)

## Optimize!
result <- nloptr( x0=theta0,eval_f=objfun,eval_grad_f=gradient,opts=options,y=y,X=X)
print(result)
betahat  <- result$solution[1:(length(result$solution)-1)]
sigmahat <- result$solution[length(result$solution)]

## Check solution
print(summary(lm(Sepal.Length~Sepal.Width+Petal.Length+Petal.Width+Species,data=iris)))


#----------------------------------
# Batch gradient descent with OLS
#----------------------------------

# set up a stepsize
alpha <- 0.00003

# set up a number of iterations
maxiter <- 500000

## Our objective function
objfun <- function(beta,y,X) {
return ( sum((y-X%*%beta)^2) )
}

# define the gradient of our objective function
gradient <- function(beta,y,X) {
return ( as.vector(-2*t(X)%*%(y-X%*%beta)) )
}

## read in the data
y <- iris$Sepal.Length
X <- model.matrix(~Sepal.Width+Petal.Length+Petal.Width+Species,iris)

## initial values
beta <- runif(dim(X)[2]) #start at uniform random numbers equal to number of coefficients

# randomly initialize a value to beta
set.seed(100)

# create a vector to contain all beta's for all steps
beta.All <- matrix("numeric",length(beta),maxiter)

# gradient descent method to find the minimum
iter  <- 1
beta0 <- 0*beta
while (norm(as.matrix(beta0)-as.matrix(beta))>1e-8) {
    beta0 <- beta
    beta <- beta0 - alpha*gradient(beta0,y,X)
    beta.All[,iter] <- beta
    if (iter%%10000==0) {
        print(beta)
    }
    iter <- iter+1
}

# print result and plot all xs for every iteration
print(iter)
print(paste("The minimum of f(beta,y,X) is ", beta, sep = ""))

## Closed-form solution
print(summary(lm(Sepal.Length~Sepal.Width+Petal.Length+Petal.Width+Species,data=iris)))




#----------------------------------
# Stochastic gradient descent with OLS
#----------------------------------
# set up parameters
alpha <- 0.0003
max_iter <- 1000000

## Our objective function
objfun <- function(beta, y, X) {
    return(sum((y - X %*% beta)^2))
}

# define the gradient for a single observation
gradient <- function(beta, y_i, X_i) {
    # For a single observation: -2 * X_i^T * (y_i - X_i*beta)
    return(-2 * t(X_i) * (y_i - sum(X_i * beta)))
}

## read in the data
y <- iris$Sepal.Length
X <- model.matrix(~Sepal.Width+Petal.Length+Petal.Width+Species, iris)

## initial values
set.seed(100)
beta <- runif(ncol(X))

# Storage for beta history
beta_history <- matrix(0, nrow=ncol(X), ncol=max_iter)
beta_history[,1] <- beta

# stochastic gradient descent method
iter <- 1
converged <- FALSE
beta_prev <- beta + 1  # Ensure different from beta initially

while (!converged && iter < max_iter) {
    # Store previous beta
    beta_prev <- beta
    
    # Randomly re-order the data
    random <- sample(nrow(X))
    X_shuffled <- X[random,]
    y_shuffled <- y[random]
    
    # Update parameters for each row of data
    for(i in 1:nrow(X)) {
        X_i <- as.numeric(X_shuffled[i,])  # Convert to numeric vector
        y_i <- y_shuffled[i]
        grad <- gradient(beta, y_i, X_i)
        beta <- beta - alpha * grad
    }
    
    # Reduce learning rate
    alpha <- alpha / 1.0005
    
    # Store current beta
    if (iter < max_iter) {
        beta_history[, iter+1] <- beta
    }
    
    # Check convergence
    if (sqrt(sum((beta - beta_prev)^2)) < 1e-12) {
        converged <- TRUE
    }
    
    # Print progress
    if (iter %% 1000 == 0) {
        cat("Iteration:", iter, "Beta:", beta, "\n")
    }
    
    iter <- iter + 1
}



# Print results
cat("Total iterations:", iter - 1, "\n")
cat("Final beta values:", beta, "\n")

# Compare with closed-form solution
ols_model <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width + Species, data=iris)
cat("\nClosed-form solution:\n")
print(coef(ols_model))
