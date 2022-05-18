#####Package#####
{
  library(readxl)
  library(dplyr)
  library(tseries)
}
#####Setting#####
setwd("C:/RRR/")
stockfuture <- read_excel("完整資料.xlsx")
summary(stockfuture)

stockfuture_train <- stockfuture[1:351,]
stockfuture_test  <- stockfuture[352:475,]

summary(stockfuture_train)
summary(stockfuture_test)

sd(stockfuture_train$nextday_openchange)
sd(stockfuture_test$nextday_openchange)

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

plot(stockfuture$trend,stockfuture$today_openchange)

#####ADF檢定#####
#確認我們的X和Y是否都為I(0)
# adf.test(stockfuture$nextday_openchange)
# adf.test(stockfuture$today_indaychange)
# adf.test(stockfuture$today_volume)
# adf.test(stockfuture$dealer)
# adf.test(stockfuture$trust_invest)
# adf.test(stockfuture$foreign_invest)
# adf.test(stockfuture$SP500change)
# adf.test(stockfuture$SP500volume)
# adf.test(stockfuture$interestchange)
# adf.test(stockfuture$USDchange)
# adf.test(stockfuture$VIXchange)
# adf.test(stockfuture$VIXvolume)
# adf.test(stockfuture$bitchange)
# adf.test(stockfuture$bitvolume)
# adf.test(stockfuture$ETHchange)
# adf.test(stockfuture$ETHvolume)
# adf.test(stockfuture$covidchange)
#都為I(0)

#####Without feature selecting#####
stockfuture_OLS <- lm(data = stockfuture_train, nextday_openchange ~ . - date)
summary(stockfuture_OLS)
#R^2為0.5102，OLS樣本內MSE為0.755

stockfuture.OLS.pred.test  <- predict(stockfuture_OLS, newdata = stockfuture_test)
rmse( actual = stockfuture.OLS.pred.test, predicted =  stockfuture_test$nextday_openchange)
#OLS樣本外MSE為0.731

#同場加映，台指期與美股的相對關係
summary(lm(SP500change ~ . , data = stockfuture_train))
summary(lm(SP500change ~ today_openchange + today_indaychange + today_volume, data = stockfuture_train))
summary(lm(nextday_openchange ~ SP500change , data = stockfuture_train))
#台股受美股影響比較大

#####PCA#####
PCA <- read.csv("PCA.csv")
stockfuture_PCA <- cbind( nextday_openchange =  stockfuture$nextday_openchange , PCA)

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

stockfuture_PCA_OLS <- lm(data = stockfuture_PCA_train , nextday_openchange ~ .)
summary(stockfuture_PCA_OLS)
#R^2為0.3612，OLS樣本內MSE為0.825

stockfuture.PCA.OLS.pred.test  <- predict(stockfuture_PCA_OLS, newdata = stockfuture_PCA_test)
rmse(stockfuture.PCA.OLS.pred.test , stockfuture_PCA_test$nextday_openchange)
#PCA的OLS之樣本外MSE為0.604

#####LASSO#####
stockfuture %>%
  dplyr::select(nextday_openchange,openchange_lag1,SP500change,SP500change_lag4,USDchange,VIXchange,bitchange) -> stockfuture_LASSO

stockfuture_LASSO_train <- stockfuture_LASSO[1:351,]
stockfuture_LASSO_test  <- stockfuture_LASSO[352:475,]

stockfuture_LASSO_OLS <- lm(data = stockfuture_LASSO_train , nextday_openchange ~ .)
summary(stockfuture_LASSO_OLS)
#R^2為0.4391，LASSO樣本內MSE為0.773

stockfuture.LASSO.OLS.pred.test  <- predict(stockfuture_LASSO_OLS, newdata = stockfuture_LASSO_test)
rmse(stockfuture.LASSO.OLS.pred.test , stockfuture_LASSO_test$nextday_openchange)
#LASSO的OLS之樣本外MSE為0.504

#####Random ForestF#####
stockfuture %>%
  dplyr::select(nextday_openchange,VIXchange,SP500change,SP500volume,today_volume,indaychange_lag2,VIXvolume) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

stockfuture_RF_OLS <- lm(data = stockfuture_RF_train , nextday_openchange ~ .)
summary(stockfuture_RF_OLS)
#R^2為0.3448，RF樣本內MSE為0.836

stockfuture.RF.OLS.pred.test  <- predict(stockfuture_RF_OLS, newdata = stockfuture_RF_test)
rmse(stockfuture.RF.OLS.pred.test , stockfuture_RF_test$nextday_openchange)
#RF樣本外MSE為0.508

#####LASSO和RF的交集變數#####
stockfuture %>%
  dplyr::select(nextday_openchange,SP500change,VIXchange) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

stockfuture_core_OLS <- lm(data = stockfuture_core_train , nextday_openchange ~ .)
summary(stockfuture_core_OLS)
#R^2為0.3229，OLS樣本內MSE為0.845

stockfuture.core.OLS.pred.test  <- predict(stockfuture_core_OLS, newdata = stockfuture_core_test)
rmse(stockfuture.core.OLS.pred.test , stockfuture_core_test$nextday_openchange)
#core的OLS之樣本外MSE為0.498

#####OLS小結#####
#在沒有特徵選取的情況下，OLS的預測能力非常差，甚至比樣本平均表現來的差
#而在特徵選取後，OLS的預測能力增加蠻多的，尤其在LASSO和Random Forest取到的變數交集，樣本外MSE可以達到0.498
