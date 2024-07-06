# Install and load Lib
pacman::p_load(rio, ggplot2)

setwd("C:/UEL/RPython/Practice_R")

# Load data
dat <- import("./data/Income.csv")
head(dat)

X <- dat[,1]
y <- dat[,2]

# Explore the relationship bêween X and y with a scatter plot
plot(X,y)

# Create model
model <- lm(y ~ X)

# Print model summary
summary(model)

# Plot the data and the regression line
plot(X, y, main = "Linear Regrssion Example", xlab = "Income", ylab = "Expenditure")
abline (model,col = "red")

# Calculate residuals
residuals <- resid(model)

# Calculate Mean Square Error (MSE)
mse <- mean(residuals^2)

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mse)

# Print residuals, MSE, and RMSE
print("Residuals: ")
print(residuals)
cat("\n")
print(paste("Mean Squared Error (MSE):", mse))
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Make predictions for new data
new_data <- data.frame(X = 26:32)
predicted_values <- predict(model, newdata = new_data)

# Print predicted values
print(predicted_values)
