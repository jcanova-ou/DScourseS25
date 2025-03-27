library(tidyverse)

test <- function() {
set.seed(100)
N <- 100000
K <- 10
sigma <- 0.5
X <- matrix(rnorm(N*K,mean=0,sd=sigma),N,K)
X[,1] <- 1 # first column of X should be all ones
eps <- rnorm(N,mean=0,sd=0.5)
betaTrue <- as.vector(runif(K))
Y <- X%*%betaTrue + eps
browser()
estimates <- lm(Y~X -1)
print(summary(estimates))
print(betaTrue)
}

test()
