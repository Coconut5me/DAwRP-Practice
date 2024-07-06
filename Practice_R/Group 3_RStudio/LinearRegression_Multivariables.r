#Install and load packages
pacman::p_load(pacman,rio,ggplot2)

#Setting working direcctory 
setwd("E:/R")

# Load data
data <- import("./data/df_processed.csv")


set.seed(123)
train_set <- sample(nrow(data), 0.7 * nrow(data))
test_set <- setdiff(1:nrow(data), train_set)

training <- data[train_set, ]
testing <- data[test_set, ]


#Create model
# model <- lm(y ~ X +X1 +X2)
lm_model <- lm(charges ~ scaled_age + scaled_bmi + is_smoker, data = training)


# R-squared
rsq_train <- summary(lm_model)$r.squared

# RMSE
rmse_train <- sqrt(mean((predict(lm_model, training) - training$charges)^2))


predictions <- predict(lm_model, newdata = testing)

plot(testing$charges, predictions, main = "Actual vs Predicted Charges", 
     xlab = "Actual Charges", ylab = "Predicted Charges", col = "blue")

abline(0, 1, col = "red")
legend("topright", legend = c("Predictions", "Perfect Prediction"), 
       col = c("blue", "red"), pch = 1)


