library(rootoned)

# Experiment 1
# data generated with different time-dependent effects
## hazard
### Here I changed the hazard function
# dataset 0




# 
# h <- function(t, x1, x2, x3, x4, x5, base_hazard_function){
#   base_hazard_function(t) * exp((-0.9 + 0.001 * t + 0.9 * log(t)) * x1
#                                 - 0.5 * x2
#                                 - 5e-2 * x3
#                                 + 1e-6 * x4
#                                 + 0.001 * x3*x4
#                                 + 1e-10 * x5)
# }


h <- function(t, x1, x2, x3, x4, x5, base_hazard_function){
  base_hazard_function(t) * exp(+0.01* x1 
                                + 0.1 * x2
                                + 0.01 * x3
                                + 0.15 * x4 
                                - 1e-20 * x5
                                - 0.005*x3*x4
  )
}
## CHF
inth <- function(t, x1, x2, x3, x4, x5, base_hazard_function){
  as.numeric(integrate(h, 0, t, x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, base_hazard_function=base_hazard_function)[1])
}

## SF
S <- function(t, x1, x2, x3, x4, x5, base_hazard_function){
  exp(-inth(t, x1, x2, x3, x4, x5, base_hazard_function))
}



generate_dataset <- function(base_hazard_function, seed=42){
    set.seed(seed)
    data <- data.frame()
    survs <- matrix(nrow=0, ncol=100)
    x1 <- rbinom(1000, 1, 0.5)
    x2 <- rbinom(1000, 1, 0.5)
    x3 <- rnorm(1000, 10, 2)
    x4 <- rnorm(1000, 20, 4)
    x5 <- rnorm(1000, 0, 1)
    X = data.frame(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5)
    generated_t <- numeric(1000)
    survs <- matrix(nrow=1000, ncol=100)
    hazards <- matrix(nrow=1000, ncol=100)
    to_drop <- numeric(0)
    for (i in 1:1000){
      u <- runif(1)
      g <- function(t, x1, x2, x3, x4, x5, base_hazard_function){
        S(t, x1, x2, x3, x4, x5, base_hazard_function) - u
      }
      skip_to_next <- FALSE
      tryCatch({
        generated_t[i] <- brent(g, 1e-16, 20, x1=x1[i], x2=x2[i], x3=x3[i], x4=x4[i], x5=x5[i], base_hazard_function=base_hazard_function)$root
      },
      error=function(cond){
        message(i)
        to_drop <<- c(to_drop, i)
        skip_to_next <<- TRUE}
      )
      if(skip_to_next) next
      # survs[i, ] <- sapply(seq(1e-9, 16, length.out=100), S, x1[i], x2[i], x3[i], x4[i], x5[i])
      # hazards[i, ] <- sapply(seq(1e-9, 16, length.out=100), h, x1[i], x2[i], x3[i], x4[i], x5[i])
    }
    
    cl <- runif(1000, 11, 16)
    cr <- runif(1000, 0, 24)
    X$time <- pmin(generated_t, cl, cr)
    X$event <- as.numeric(generated_t < pmin(cl, cr))
    X
  }
  



## baseline hazard function -- EXP1_complex
h0 <- function(t){
  exp(-17.8 + 6.5 * t - 11 * t ^ 0.5 * log(t) + 9.5 * t ^ 0.5)
}

X <- generate_dataset(h0)

X

# event/censoring times histogram
c1 <- rgb(173,216,230,max = 255, alpha = 80, names = "lt.blue")
c2 <- rgb(255,192,203, max = 255, alpha = 80, names = "lt.pink")

# histogram_simulated_event_times
hist(X$time[X$event==1], col=c1, xlab= "Time", ylab="Event Frequency", main="Histogram of Event Times")
hist(X$time[X$event==0], col=c2, add=TRUE, breaks=0:16)
legend("topright", c("uncensored", "censored"), col=c(c1, c2), lwd=10)

# number of events vs x1 variable
table(cut(X$time[X$event==1], breaks = c(0, 2, 4, 6, 8, 10, 12, 16)), X$x1[X$event==1])

# survial_curve_simulated_data
# https://cran.r-project.org/web/packages/ggfortify/vignettes/plot_surv.html
library(ggfortify)
library(survival)
autoplot(survival::survfit(survival::Surv(time, event) ~ 1, data = X), surv.colour = 'orange', censor.colour = 'red', ylab = "Survival Probability", xlab="Time")


# survival curves
fit <- survival::survfit(survival::Surv(time, event)~x1, data=X)
survminer::ggsurvplot(fit, data=X, pval = TRUE)
## Here I changed the filepath
write.csv(X, "/Users/abdallah/Desktop/Kings College Project/Code/exp1_data_complexADAPTED2.csv", row.names = FALSE)



