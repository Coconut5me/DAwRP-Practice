# Install and load required lib
pacman::p_load(rio, ggplot2, forecast, tseries, urca, caret)

# Ignore warning
options(warn = -1)

# Setting working directory
setwd('C:/UEL/RPython/Practice_R')

# Load data
data <- import('./data/ACG.csv')
head(data)

# Plot the Close price
ggplot(data, aes(x = Date, y = Close)) + geom_line() + labs(x = "Date", y = "Close Price", title = "Close Price Over Time")

# Check the Stationary
check_stationary <- function(data, significance_level) {
  # Perform ADF test (null hypothesis: unit root, non-stationary)
  adf_test <- adf.test(data)
  adf_pvalue <- adf_test$p.value
  
  # Perform KPSS test (null hypothesis: unit root, non-stationary)
  kpss_test <- kpss.test(data)
  kpss_pvalue <- kpss_test$p.value
  
  # Output results
  cat("ADF Test p-value:", adf_pvalue, "\n")
  cat("KPSS Test p-value:", kpss_pvalue, "\n")
  
  # Check for stationarity based on significance level
  if (adf_pvalue < significance_level) {
    cat("Series is likely stationary based on ADF test\n")
  } else {
    cat("Series might be non-stationary based on ADF test\n")
  }
  
  if (kpss_pvalue > significance_level) {
    cat("Series is likely stationary based on KPSS test\n")
  } else {
    cat("Series might be non-stationary based on KPSS test\n")
  }
}

close_price_data = log(data$Close)
significance_level <- 0.05 # Common significance level

check_stationary(close_price_data, significance_level)

close_price_data_diff = diff(close_price_data)
check_stationary(close_price_data_diff, significance_level)
plot.ts(close_price_data_diff)

# Plot PACF/ACF
pacf(close_price_data_diff, main = "PACF of Close Price")
acf(close_price_data_diff, main = "ACF of Close Price")

# Split data into training and testing sets (90% training, 10% testing)
set.seed(123)
train_size <- floor(0.9 * length(close_price_data))
train_index <- seq_len(train_size)
test_index <- (train_size + 1):length(close_price_data)  # Define test index
train_data <- close_price_data[train_index]
test_data <- close_price_data[test_index]

# Fit ARIMA model 
arima_model <- auto.arima(train_data)

# Make predictions for the next 10 time points
forecast_values <- forecast(arima_model, h = length(test_data))

# Plot the time series data
plot(close_price_data, type = "l", col = "black", xlab = "Date", ylab = "Close Price", main = "Actual vs Predicted Close Price")

# Add lines for train, test, and forecasted data
lines(train_index, train_data, col = "green")
lines(test_index, test_data, col = "red")
lines(test_index, forecast_values$mean, col = "blue")

# Add legend
legend("topleft", legend = c("Actual Train", "Actual Test", "Predicted"), col = c("green", "red", "blue"), lty = 1, cex = 0.8)

