# 这里我们使用facebook发布的时间序列自动预测程序包prophet进行预测
library(prophet)
library(dplyr)
library(readr)

# 载入数据并拟合模型
df <- read_csv("~/Documents/MyPrograming/R/TimeSeriesAnalysis/Projects/coal_production/clean_data.csv")
m <- prophet(df)
future <- make_future_dataframe(m, periods = 6, freq = 'm')
tail(future)

# 进行预测
forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
plot(m, forecast)

# 趋势分解
prophet_plot_components(m, forecast)

# 误差分析
coal.data <- read_csv("~/Documents/MyPrograming/R/TimeSeriesAnalysis/Projects/coal_production/coal_data.csv")

coal.ts <- ts(rev(coal.data$Num), start = c(2000,1), end=c(2017, 10),
              frequency = 12)
fore_true <- window(coal.ts, start=c(2009, 2), end=c(2009, 7))

fore_pred <- tail(forecast$yhat)
rmse <- sqrt(mean((fore_true-fore_pred)^2))
rmse

plot(coal, lty=1, type='b', pch=16)
fore_ts <- ts(forecast$yhat,  start = c(2000,2), end=c(2009, 7),
              frequency = 12)
lines(fore_ts, lty=1, type='l', col='blue')

fore_true <- window(coal.ts, start=c(2009, 2), end=c(2009, 7))
lines(fore_true, col="red", type='b', pch=10)
