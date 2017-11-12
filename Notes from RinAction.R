# Creating a time-series object in R
sales <- c(18, 33, 41, 7, 34, 35, 24, 25, 24, 21, 25, 20,
           22, 31, 40, 29, 25, 21, 22, 54, 31, 25, 26, 35)

tsales <- ts(sales, start=c(2003, 1), frequency=12)
plot(tsales, type='o', pch=19)

# 时间序列的属性
tsales.attributes <- list(start=start(tsales), end=end(tsales), 
                          frequency=frequency(tsales))

# 使用window截取部分数据
tsales.subset <- window(tsales, start=c(2003, 5), end=c(2004, 6))



# Smoothing and seasonal decomposition

# Simple moving averages
library(forecast)
opar <- par(no.readonly=TRUE)
par(mfrow=c(2,2))
ylim <- c(min(Nile), max(Nile))
plot(Nile, main="Raw time series")
plot(ma(Nile, 3), main="Simple Moving Averages (k=3)", ylim=ylim)
plot(ma(Nile, 7), main="Simple Moving Averages (k=7)", ylim=ylim)
plot(ma(Nile, 15), main="Simple Moving Averages (k=15)", ylim=ylim)
par(opar)


# Seasonal decomposition using stl()
plot(AirPassengers)
lAirPassengers <- log(AirPassengers)
plot(lAirPassengers, ylab="log(AirPassengers)")

# Decomposes the time series
fit <- stl(lAirPassengers, s.window="period")
plot(fit)

# 分解后的数据
# In this case, fit$time.series is based on the logged time series.
# exp(fit$time.series) converts the decomposition back to the original metric.
fit$time.series
exp(fit$time.series)

# visualize a seasonal decomposition
par(mfrow=c(2,1))
library(forecast)
monthplot(AirPassengers, xlab="", ylab="")
seasonplot(AirPassengers, year.labels="TRUE", main="")
par(opar)

# Exponential forecasting models

# A simple exponential model (also called a single exponential model) fits a time series that has a
# constant level and an irregular component at time i but has neither a trend nor a sea-
# sonal component. A double exponential model (also called a Holt exponential smoothing)
# fits a time series with both a level and a trend. Finally, a triple exponential model (also
# called a Holt-Winters exponential smoothing) fits a time series with level, trend, and sea-sonal components.

#  Simple exponential smoothing
library(forecast)
fit <- ets(nhtemp, model="ANN")
print(fit)
prediction = forecast(fit, 1)
plot(prediction, xlab="Year",
     ylab=expression(paste("Temperature (", degree*F,")")),
     main="New Haven Annual Mean Temperature")

#  The mean absolute scaled error is the most recent accuracy measure and 
# is used to compare the forecast accuracy across time series on different scales.
print(accuracy(fit))

# Holt and Holt-Winters exponential smoothing
library(forecast)
fit <- ets(log(AirPassengers), model="AAA")
print(fit)
pred <- forecast(fit,5)
plot(pred)
# Makes forecasts in the original scale
pred$mean <- exp(pred$mean)
pred$lower <- exp(pred$lower)
pred$upper <- exp(pred$upper)
p <- cbind(pred$mean, pred$lower, pred$upper)
dimnames(p)[[2]] <- c("mean", "Lo 80", "Lo 95", "Hi 80", "Hi 95")


# Automatic exponential forecasting with ets()
library(forecast)
fit <- ets(JohnsonJohnson)
print(fit)
plot(forecast(fit), main="Johnson & Johnson Forecasts",
     ylab="Quarterly Earnings (Dollars)", xlab="Time", flty=2)


# ARIMA forecasting models

# Note1: ARIMA models are designed to fit stationary time series 
# (or time series that can be made stationary).

# Note2: To summarize, ACF and PCF plots are used to determine the parameters of ARIMA
# models. Stationarity is an important assumption, and transformations 
# and differenc-ing are used to help achieve stationarity. 

# Transforming the time series and assessing stationarity
library(forecast)
library(tseries)
plot(Nile)

# 差分与平稳性检验
adf.test(Nile)
ndiffs(Nile)  # >>1

dNile <- diff(Nile)
plot(dNile)
adf.test(dNile)

# IDENTIFYING ONE OR MORE REASONABLE MODELS
Acf(dNile)
Pacf(dNile)

# Fitting an ARIMA model
fit <- arima(Nile, order=c(0,1,1))
print(fit)
print(accuracy(fit))

#  Evaluating the model fit
qqnorm(fit$residuals)
qqline(fit$residuals)
# The Box.test() function provides a test that the autocorrelations are all zero. The
# results aren’t significant, suggesting that the autocorrelations don’t differ from zero.
Box.test(fit$residuals, type="Ljung-Box")

# MAKING FORECASTS
pred <- forecast(fit, 3)
plot(pred)


# Automated ARIMA forecasting
library(forecast)
fit <- auto.arima(sunspots)
print(fit)
pred <- forecast(fit, 3)
print(pred)
print(accuracy(pred))
plot(pred)




