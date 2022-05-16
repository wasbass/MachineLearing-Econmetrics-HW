#####Setting#####
{
library(gbm)
library(readxl)
library(MASS)
library(doParallel)
library(DAAG)
library(caret)
library(dplyr)
library(tseries)
}

stockfuture <- read_excel("C:/RRR/完整資料.xlsx")
summary(stockfuture)

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

stockfuture_train <- stockfuture[1:351,]
stockfuture_test  <- stockfuture[352:475,]

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

plot(stockfuture$trend,stockfuture$today_openchange)
#####OLS#####
set.seed(1)
stockfuture_OLS <- lm(data = stockfuture_train, nextday_openchange ~ . - date)
summary(stockfuture_OLS)
#R^2為0.5102，OLS樣本內MSE為0.7549

stockfuture.OLS.pred.test  <- predict(stockfuture_OLS, newdata = stockfuture_test)
rmse(stockfuture.OLS.pred.test, stockfuture_test$nextday_openchange)
#OLS樣本外MSE為0.7308

#同場加映，台指期與美股的相對關係
summary(lm(SP500change ~ . , data = stockfuture_train))
summary(lm(SP500change ~ today_openchange + today_indaychange + today_volume, data = stockfuture_train))
summary(lm(nextday_openchange ~ SP500change , data = stockfuture_train))
#台股受美股影響比較大

#####Gradient Boosting#####

#Tuning
ctrl <- trainControl(method = "repeatedcv",number = 10, repeats = 2, allowParallel = T) #重複的cv，每次分10組，重複2次
registerDoParallel(detectCores()-1)
grid <- expand.grid(n.trees = c(5000,10000,20000), interaction.depth=c(1:3), shrinkage=c(0.05,0.01,0.001) , n.minobsinnode=c(5,10,20))

# set.seed(1)
# stockfuture.gbm.caret <- caret::train(nextday_openchange ~ . - date , data = stockfuture_train, method = "gbm", metric = "RMSE",
#                                       trControl = ctrl, tuneGrid = grid)
# print(stockfuture.gbm.caret)
#從結果來看，n.trees = 10000，shrinkage=0.001，interaction.depth=1，n.minobsinnode=5會使得樣本內的RMSE在CV下最小化，達到0.8361

#Build Model
set.seed(1)
stockfuture_gbm <- gbm(nextday_openchange ~ . - date , data = stockfuture_train,
                         distribution = "gaussian", n.trees = 10000,
                         interaction.depth = 1, shrinkage = 0.001,
                         bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)
#在Tuning，也就是優化參數的過程中，由training的RMSE說明n.minobsinnode越小越好
#但從樣本外MSE來看，n.minobsinnode過小會存在overfitting的問題
#在我們持續加大n.minobsinnode後，樣本外MSE反而越來越小
#因此我們最後選擇用小的shrinkage，大的n.trees來增強學習能力
#並且用小的interaction.depth，大的n.minobsinnode來避免過度配適問題

sum(summary(stockfuture_gbm)$rel.inf)#合計為100，代表每個變數影響力所佔的百分比
summary(stockfuture_gbm)
#可以看出各個變數的相對重要性
#在本模型中，SP500和VIX的價格變化幅度是最重要的變數，遠超其他變數
#但要注意這個重要性可能會受到shrinkage和n.trees的數值所影響

stockfuture.gbm.pred.train <- predict(stockfuture_gbm, newdata = stockfuture_train)
rmse(stockfuture.gbm.pred.train, stockfuture_train$nextday_openchange)
#Gradient Boosting樣本內MSE為0.849

stockfuture.gbm.pred.test  <- predict(stockfuture_gbm, newdata = stockfuture_test)
rmse(stockfuture.gbm.pred.test, stockfuture_test$nextday_openchange)  
#Gradient Boosting樣本外MSE為0.492

#從結果來看，不放落遲項的預測能力比較好，而且放了會有overfitting的問題


#####After feature selecting#####
#####PCA#####
PCA <- read.csv("C:/RRR/PCA.csv")
stockfuture_PCA <- cbind( nextday_openchange =  stockfuture$nextday_openchange , PCA)

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

#OLS
set.seed(1)
stockfuture_PCA_OLS <- lm(data = stockfuture_PCA_train , nextday_openchange ~ .)
summary(stockfuture_PCA_OLS)
#R^2為0.3612，OLS樣本內MSE為0.825

stockfuture.PCA.OLS.pred.test  <- predict(stockfuture_PCA_OLS, newdata = stockfuture_PCA_test)
rmse(stockfuture.PCA.OLS.pred.test , stockfuture_PCA_test$nextday_openchange)
#PCA的OLS之樣本外MSE為0.604

#Gradient Boosting
set.seed(1)
stockfuture_PCA_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_PCA_train,
                       distribution = "gaussian", n.trees = 10000,
                       interaction.depth = 1, shrinkage = 0.001,
                       bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

stockfuture.PCA.gbm.pred.test  <- predict(stockfuture_PCA_gbm, newdata = stockfuture_PCA_test)
rmse(stockfuture.PCA.gbm.pred.test, stockfuture_PCA_test$nextday_openchange)  
#PCA的Gradient Boosting樣本外MSE為0.593

#####LASSO#####
stockfuture <- read_excel("C:/RRR/完整資料.xlsx")
stockfuture %>%
  select(nextday_openchange,openchange_lag1,SP500change,SP500change_lag4,USDchange,VIXchange,bitchange) -> stockfuture_lasso

stockfuture_lasso_train <- stockfuture_lasso[1:351,]
stockfuture_lasso_test  <- stockfuture_lasso[352:475,]

#OLS
set.seed(1)
stockfuture_lasso_OLS <- lm(data = stockfuture_lasso_train , nextday_openchange ~ .)
summary(stockfuture_lasso_OLS)
#R^2為0.4391，OLS樣本內MSE為0.7731

stockfuture.lasso.OLS.pred.test  <- predict(stockfuture_lasso_OLS, newdata = stockfuture_lasso_test)
rmse(stockfuture.lasso.OLS.pred.test , stockfuture_lasso_test$nextday_openchange)
#lasso的OLS之樣本外MSE為0.504

#Gradient Boosting
set.seed(1)
stockfuture_lasso_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_lasso_train,
                           distribution = "gaussian", n.trees = 10000,
                           interaction.depth = 1, shrinkage = 0.001,
                           bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

stockfuture.lasso.gbm.pred.test  <- predict(stockfuture_lasso_gbm, newdata = stockfuture_lasso_test)
rmse(stockfuture.lasso.gbm.pred.test, stockfuture_lasso_test$nextday_openchange)  
#lasso的Gradient Boosting樣本外MSE為0.496

#####RF#####
#stockfuture <- read_excel("C:/RRR/完整資料.xlsx")
stockfuture %>%
  select(nextday_openchange,VIXchange,SP500change,SP500volume,today_volume,indaychange_lag2,VIXvolume) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

#OLS
stockfuture_RF_OLS <- lm(data = stockfuture_RF_train , nextday_openchange ~ .)
summary(stockfuture_RF_OLS)
#R^2為0.3448，OLS樣本內MSE為0.8355

stockfuture.RF.OLS.pred.test  <- predict(stockfuture_RF_OLS, newdata = stockfuture_RF_test)
rmse(stockfuture.RF.OLS.pred.test , stockfuture_RF_test$nextday_openchange)
#RF的OLS之樣本外MSE為0.508

#Gradient Boosting
set.seed(1)
stockfuture_RF_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_RF_train,
                             distribution = "gaussian", n.trees = 10000,
                             interaction.depth = 1, shrinkage = 0.001,
                             bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

stockfuture.RF.gbm.pred.test  <- predict(stockfuture_RF_gbm, newdata = stockfuture_RF_test)
rmse(stockfuture.RF.gbm.pred.test, stockfuture_RF_test$nextday_openchange)  
#RF的Gradient Boosting樣本外MSE為0.486，比原本的好

#加碼再看看樣本內的MSE
stockfuture.RF.gbm.pred.train  <- predict(stockfuture_RF_gbm, newdata = stockfuture_RF_train)
rmse(stockfuture.RF.gbm.pred.train, stockfuture_RF_train$nextday_openchange)  
#RF的Gradient Boosting樣本內MSE為0.881，比原本的差
#代表我們在篩選特徵後，稍微解決了Gradient Boosting原本有的Overfitting問題

#####LASSO和RF的交集變數#####
#只有SP500change和VIXchange
stockfuture %>%
  select(nextday_openchange,SP500change,VIXchange) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]


#OLS
stockfuture_core_OLS <- lm(data = stockfuture_core_train , nextday_openchange ~ .)
summary(stockfuture_core_OLS)
#R^2為0.3229，OLS樣本內MSE為0.8445

stockfuture.core.OLS.pred.test  <- predict(stockfuture_core_OLS, newdata = stockfuture_core_test)
rmse(stockfuture.core.OLS.pred.test , stockfuture_core_test$nextday_openchange)
#core的OLS之樣本外MSE為0.498

#Gradient Boosting
set.seed(1)
stockfuture_core_gbm <- gbm(nextday_openchange ~ . , data = stockfuture_core_train,
                          distribution = "gaussian", n.trees = 10000,
                          interaction.depth = 1, shrinkage = 0.001,
                          bag.fraction = 0.5, cv.folds = 10,n.minobsinnode=50)

stockfuture.core.gbm.pred.test  <- predict(stockfuture_core_gbm, newdata = stockfuture_core_test)
rmse(stockfuture.core.gbm.pred.test, stockfuture_core_test$nextday_openchange)  
#core的Gradient Boosting樣本外MSE為0.490，比原本的好

#####結論#####
#不做篩選的Gradient Boosting的樣本外預測已經蠻不錯的了
#但是經過Random Forest篩選變數後再做Gradient Boosting，樣本外預測的結果會更好
#相較之下，PCA和LASSO的篩選變數結果就不那麼優秀
#而若是拿LASSO和Random Forest都有選到的變數做預測，結果也不錯