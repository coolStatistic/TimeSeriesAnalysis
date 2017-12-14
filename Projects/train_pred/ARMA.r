# 通过ARMA模型分析全国的原煤产量
# 数据来源：http://data.stats.gov.cn/easyquery.htm?cn=A01
# 通过Python进行了预处理，选取当期产量作为指标
library(forecast)
library(tseries)

library(readr)
coal.data <- read_csv("~/Documents/MyPrograming/R/TimeSeriesAnalysis/Projects/coal_production/coal_data.csv")

coal.ts <- ts(rev(coal.data$Num), start = c(2000,1), end=c(2017, 10),
              frequency = 12)

# 选取九年的数据
coal <- window(coal.ts, start=c(2000, 1), end=c(2009, 1))
head(coal, 12)

# 通过时序图初步观察
plot(coal)

# 通过上图，很显然其不是平稳的。同时，我们使用ADF进行检验，也可得到其不是平稳的。
# ADF检验
library(aTSA)
stationary.test(coal, method = "adf")

# 所以，我们进行差分，并对差分后的序列再次检验
diff_n = ndiffs(coal)  # 简单识别需要差分的阶数
coal_dif = diff(coal)
plot(coal_dif)
stationary.test(coal_dif, method = "adf")

# 可以看到，此时已经平稳。亦即满足了ARMA建模的条件
# 下面，我们使用Box-Jenkins的方法进行建模
#　模型识别，acf与pacf

Acf(coal_dif)
Pacf(coal_dif)

# 进一步做季节差分
coal_difDif <- diff(coal_dif, 12)  # d=1 D=1
Acf(coal_difDif)  # q=0 Q=1
Pacf(coal_difDif)  # p=1 P=2 

fit1 <- Arima(coal, order=c(1,1,1),seasonal=c(1,1,1))
fit2 <- Arima(coal, order=c(1,1,1),seasonal=c(1,1,2))
fit3 <- Arima(coal, order=c(1,1,1),seasonal=c(1,1,3))

plot(fit1)

# 模型已经建立，下面进行模型的检验
# 残差白噪声检验(Box-Pierce test)
Box.test(fit1$residuals, lag = 6)
Box.test(fit1$residuals, lag = 12)

# 当然，也可以使用Box-Ljung test
Box.test(fit1$residuals, type = "Ljung-Box", lag=6)
Box.test(fit1$residuals, type = "Ljung-Box", lag=12)


# 可以看到已经通过了白噪声检验，意味着拟合效果比较好
# 下面，我们作出拟合的效果图
plot(fit1$fitted, col='blue', lty=1, lwd=2)
lines(coal, col='black', lty=1, type='b', lwd=0.5, pch=16)
legend(2000, 24000, 
       c("Fitted", "True Value"),
       lty=c(1,2),
       pch = c(NA, 16),
       col=c("blue", "black"))

#可以看到，拟合效果已经是非常地好了。
# 我们还可以查看更多关于精度的信息
accuracy(fit1)

# 最后，我们对后面六个月的数据进行预测
fore <- forecast::forecast(fit1, h=6)
plot(fore)
print(fore)

# 由于我们是有后面的真实数据的，所以我们放进去来作下对比
fore_true <- window(coal.ts, start=c(2009, 2), end=c(2009, 7))
plot(fore)
lines(fore_true, col="red", type='b', pch=10)


# 具体看下残差
plot(fore$residuals)
fore_true
fore$mean 
RMSE = sqrt(mean((fore_true-fore$mean)^2))

