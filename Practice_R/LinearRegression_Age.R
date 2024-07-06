# Install and load Lib
pacman::p_load(rio, ggplot2)

setwd("C:/UEL/RPython/Practice_R")

# Load data
dat <- import("./data/df_processed.csv")
head(dat)

X <- dat[,2]
y <- dat[,4]

# Explore the relationship bêween X and y with a scatter plot
plot(X,y)

# Create model
model <- lm(y ~ X)

# Print model summary
summary(model)

# Plot the data and the regression line
plot(X, y, main = "Linear Regrssion Example", xlab = "Scaled Age", ylab = "Charges")
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
new_age <- 26:32
scaled_X <- scale(new_age, center = mean(dat$X), scale = sd(dat$X))
print(scaled_X)

new_data <- data.frame(X = scaled_X)
predicted_values <- predict(model, newdata = new_data)

# Print predicted values
print(predicted_values)
