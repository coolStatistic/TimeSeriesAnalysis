
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
#　模型识别，ａｃｆ与ｐａｃｆ

Acf(coal_dif)
Pacf(coal_dif)

# 进一步做季节差分
coal_difDif <- diff(coal_dif, 12)  # d=1 D=1
Acf(coal_difDif)  # q=0 Q=1
Pacf(coal_difDif)  # p=1 P=2 

fit1 <- arima(coal, order=c(1,1,0),seasonal=c(2,1,1))

# 发现都是拖尾的。所以应该为ARMA模型
# 这块手动识别的理论还没有理解，直接用auto吧先...
# arima
fit <- auto.arima(coal)
plot(fit)
fit

# 模型已经建立，下面进行模型的检验
# 残差白噪声检验(Box-Pierce test)
Box.test(fit$residuals)

# 当然，也可以使用Box-Ljung test
Box.test(fit$residuals, type = "Ljung-Box")

# 可以看到已经通过了白噪声检验，意味着拟合效果比较好
# 下面，我们作出拟合的效果图
plot(fit$fitted, col='green', lwd=1)
lines(coal, col='black', lwd=1.5)
legend(2000, 24000, 
       c("Fitted", "True Value"),
       lty=c(1,1),
       col=c("green", "black"))

# 可以看到，拟合效果已经是非常地好了。
# 我们还可以查看更多关于精度的信息
accuracy(fit)

# 最后，我们对后面六个月的数据进行预测
# forecats
fore <- forecast::forecast(fit, h=6)
plot(fore)
print(fore)

# 看起来一切都好的样子...
# 由于我们是有后面的真实数据的，所以我们放进去来作下对比
fore_true <- window(coal.ts, start=c(2009, 2), end=c(2009, 7))
plot(fore)
lines(fore_true, col="red")

# 发现差别还是比较大的...
# 我们来具体看下残差
# plot(fore$residuals)
fore_true
fore$mean

RMSE = sqrt(mean((fore_true-fore$mean)^2))


