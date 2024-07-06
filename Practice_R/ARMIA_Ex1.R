# Import Lib
pacman::p_load(rio,ggplot2,forecast,tseries,urca)

# Ignore warning
options(warn=-1)

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
significance_level <- 0.05 #Common significance level

check_stationary(close_price_data,significance_level)

close_price_data_diff = diff(close_price_data)
check_stationary(close_price_data_diff,significance_level)
plot.ts(close_price_data_diff)

# Plot PACF/ACF
pacf(close_price_data_diff, main="PACF of Close Price")
acf(close_price_data_diff, main="ACF of Close Price")

# Fit ARIMA model 
arima_model <- auto.arima(close_price_data)
# arima_model <- auto.arima(close_price_data_diff)

#=================================================
# Summary of the fitted model
summary(arima_model)

# Plot the time series data and forecast
plot(forecast(arima_model))

# Evaluate model performance 
accuracy(arima_model)

# Make predictions for the next 90 time points
forecast_values <- forecast(arima_model, h=90)

# Print the forecasted values
# print(forecast_values)
plot(forecast_values)

#=================================================

# Fit model
model = arima(close_price_data, order=c(4,1,3), seasonal = (list(order=c(4,1,3))))
model
help(arima)
help(auto.arima)

resid_ = residuals(model)
plot.ts(resid_)
gghistogram(resid_)

# Make predictions for the next 10 time points
# forecast_values <- forecast(model, h=90)
forecast_values <- forecast(model, xreg = 1:(length))

plot(forecast_values)



       
       
