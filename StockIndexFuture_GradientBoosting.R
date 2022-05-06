library(gbm)
library(readxl)
library(MASS)

stockfuture <- read_excel("C:/RRR/完整資料.xlsx")
summary(stockfuture)

stockfuture_train <- stockfuture[1:330,]
stockfuture_test  <- stockfuture[331:475,]

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

#####OLS#####
stockfuture_OLS <- lm(nextday_openchange ~ . - date , data = stockfuture_train)
summary(stockfuture_OLS)
#R^2為0.4339，OLS樣本內MSE為0.8058

stockfuture.OLS.pred.test  <- predict(stockfuture_OLS, newdata = stockfuture_test)
rmse(stockfuture.OLS.pred.test, stockfuture_test$nextday_openchange)
#OLS樣本外MSE為0.5603

#同場加映，台指期與美股的相對關係
summary(lm(SP500change ~ . , data = stockfuture_train))
summary(lm(SP500change ~ today_openchange + today_indaychange + today_volume, data = stockfuture_train))
summary(lm(nextday_openchange ~ SP500change , data = stockfuture_train))
#台股受美股影響比較大

#####Gradient Boosting#####
set.seed(1)
stockfuture_gbm_1 <- gbm(nextday_openchange ~ . - date , data = stockfuture_train,
                         distribution = "gaussian", n.trees = 10000,
                         interaction.depth = 1, shrinkage = 0.001,
                         bag.fraction = 0.5, cv.folds = 10)
#原本有設定n.minobsinnode=5，意即在模型中最少使用到5個變數
#但從樣本外MSE來看，會存在overfitting的問題，決定拿掉


sum(summary(stockfuture_gbm_1)$rel.inf)#合計為100，代表每個變數影響力所佔的百分比
summary(stockfuture_gbm_1)
#可以看出各個變數的相對重要性
#但這個重要性很容易受到shrinkage和n.trees的數值所影響

stockfuture.gbm.1.pred.train <- predict(stockfuture_gbm_1, newdata = stockfuture_train)
rmse(stockfuture.gbm.1.pred.train, stockfuture_train$nextday_openchange)
#Gradient Boosting樣本內MSE為0.7372

stockfuture.gbm.1.pred.test  <- predict(stockfuture_gbm_1, newdata = stockfuture_test)
rmse(stockfuture.gbm.1.pred.test, stockfuture_test$nextday_openchange)  
#Gradient Boosting樣本外MSE為0.5301

