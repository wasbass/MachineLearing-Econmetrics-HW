{
library(np)
library(readxl)
library(locfit)
library(dplyr)
library(PLRModels)
}
setwd("C:/RRR/")
stockfuture <- read_excel("完整資料.xlsx")

rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

#####PCA#####
PCA <- read.csv("C:/RRR/PCA.csv")

stockfuture_PCA <- cbind( nextday_openchange =  stockfuture$nextday_openchange , PCA)

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

sd(stockfuture_PCA_train$nextday_openchange)
sd(stockfuture_PCA_test$nextday_openchange)

#Tuning
# alpha_PCA.cv <-gcvplot(nextday_openchange ~
#           PC1+PC2 +PC3 +PC4 +PC5 +PC6 ,
#         kern = "gauss",maxk = 10000 ,deg = 1 ,data = stockfuture_PCA_train, 
#         alpha = seq(0.3,0.95,by=0.05) , df = 2)
# 
# plot(alpha_PCA.cv)
#看起來alpha取0.9會最好

#下面兩種方式其實是一樣的

LL_PCA_fit_aux   <- locfit(nextday_openchange ~
                             PC1+PC2 +PC3 +PC4 +PC5 +PC6 ,
                           kern = "gauss",maxk = 10000 ,deg = 1 ,data = stockfuture_PCA_train , alpha =  c(0.9,NULL) )

LL_PCA_fit <- locfit.raw(y = stockfuture_PCA_train$nextday_openchange ,
                         x = as.matrix(stockfuture_PCA_train[,2:7]),
                         kern = "gauss",maxk = 10000 ,deg = 1 , 
                         alpha = c(0.9,NULL))


#deg = 1代表local linear,kern代表用的kernal function
#alpha第一個參數是拿最近的幾個data當bandwidth當參照，第二個參數是用固定的歐式距離當bandwidth
#因為我們的自變數很多，而且資料算稀疏，所以bandwidth應該要設大一點，這邊是選擇0.9
#maxk是預設保留的記憶體大小，當變數比較多的時候要預留多一點

LL_PCA_pred_train <-predict(LL_PCA_fit,newdata = as.matrix(stockfuture_PCA_train[,2:7]))
rmse(actual = stockfuture_PCA_train$nextday_openchange ,predicted =  LL_PCA_pred_train)
#樣本內MSE為0.737

LL_PCA_pred_test <-predict(LL_PCA_fit, newdata = as.matrix(stockfuture_PCA_test[,2:7]))
rmse(actual = stockfuture_PCA_test$nextday_openchange ,
     predicted =  LL_PCA_pred_test)
#樣本外MSE為0.564，比Gradient Boosting還要好

#See how it fitting
# smoothingSpline_LL_PCA = smooth.spline(stockfuture_PCA_test$PC1, LL_PCA_pred_test, spar=0.5)
# 
# plot(stockfuture_PCA_test$PC1 , stockfuture_PCA_test$nextday_openchange)
# 
# lines(smoothingSpline_LL_PCA , col=4 ,lwd = 4)

#####LASSO#####
stockfuture <- read_excel("完整資料.xlsx")
stockfuture %>%
  dplyr::select(nextday_openchange,
         openchange_lag1,
         SP500change,
         SP500change_lag4,
         USDchange,
         VIXchange,
         bitchange) -> stockfuture_LASSO

stockfuture_LASSO_train <- stockfuture_LASSO[1:351,]
stockfuture_LASSO_test  <- stockfuture_LASSO[352:475,]

#Tuning
# alpha_LASSO.cv <-gcvplot(nextday_openchange ~
#                           openchange_lag1 + SP500change + SP500change_lag4 + USDchange + VIXchange + bitchange ,
#                        kern = "gauss",maxk = 10000 ,deg = 1 ,data = stockfuture_LASSO_train, 
#                        alpha = seq(0.3,0.95,by=0.05) , df = 2)
# 
# plot(alpha_LASSO.cv)
#看起來alpha取0.9和0.95差不多，那就取0.9比較能抓到local的趨勢

LL_LASSO_fit <- locfit.raw(y = stockfuture_LASSO_train$nextday_openchange ,
                         x = as.matrix(stockfuture_LASSO_train[,2:7]),
                         kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.9,NULL))

LL_LASSO_pred_train <-predict(LL_LASSO_fit,newdata = as.matrix(stockfuture_LASSO_train[,2:7]))
rmse(actual = stockfuture_LASSO_train$nextday_openchange ,predicted =  LL_LASSO_pred_train)
#樣本內MSE為0.702

LL_LASSO_pred_test <-predict(LL_LASSO_fit,newdata = as.matrix(stockfuture_LASSO_test[,2:7]))
rmse(actual = stockfuture_LASSO_test$nextday_openchange ,predicted =  LL_LASSO_pred_test)
#樣本外MSE為0.496

#See how it fitting
smoothingSpline_LL_LASSO = smooth.spline(stockfuture_LASSO_test$SP500change, LL_LASSO_pred_test, spar=0.5)

plot(stockfuture_LASSO_test$SP500change , stockfuture_LASSO_test$nextday_openchange)

lines(smoothingSpline_LL_LASSO , col=4 ,lwd = 4)

#####RF#####

stockfuture %>%
  dplyr::select(nextday_openchange,
                VIXchange,
                SP500change,
                SP500volume,
                today_volume,
                indaychange_lag2,
                VIXvolume) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

#Tuning
# alpha_RF.cv <-gcvplot(nextday_openchange ~
#                         VIXchange + SP500change + SP500volume + today_volume + indaychange_lag2 + VIXvolume ,
#                          kern = "gauss",maxk = 10000 ,deg = 1 ,data = stockfuture_RF_train, 
#                          alpha = seq(0.3,1,by=0.05) , df = 2)
# 
# plot(alpha_RF.cv)
#看起來alpha取1最好，但這樣就不local了，改取0.9

LL_RF_fit <- locfit.raw(y = stockfuture_RF_train$nextday_openchange ,
                           x = as.matrix(stockfuture_RF_train[,2:7]),
                           kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.9,NULL))

LL_RF_pred_train <-predict(LL_RF_fit,newdata = as.matrix(stockfuture_RF_train[,2:7]))
rmse(actual = stockfuture_RF_train$nextday_openchange ,predicted =  LL_RF_pred_train)
#樣本內MSE為0.788

LL_RF_pred_test <-predict(LL_RF_fit,newdata = as.matrix(stockfuture_RF_test[,2:7]))
rmse(actual = stockfuture_RF_test$nextday_openchange ,predicted =  LL_RF_pred_test)
#樣本外MSE為0.495

#See how it fitting
smoothingSpline_LL_RF = smooth.spline(stockfuture_RF_test$SP500change, LL_RF_pred_test, spar=0.5)

plot(stockfuture_RF_test$SP500change , stockfuture_RF_test$nextday_openchange)

lines(smoothingSpline_LL_RF , col=6 ,lwd = 4)

#####LASSO和RF的交集變數#####
stockfuture %>%
  dplyr::select(nextday_openchange,SP500change,VIXchange) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

#Tuning
# alpha_core.cv <-gcvplot(nextday_openchange ~
#                         VIXchange + SP500change ,
#                       kern = "gauss",maxk = 10000 ,deg = 1 ,data = stockfuture_core_train, 
#                       alpha = seq(0.3,0.95,by=0.05) , df = 2)
# 
# plot(alpha_core.cv)
#看起來0.45最好，代表如果變數取少一點，我們的bandwidth就可以選小一點，增加配適度

LL_core_fit <- locfit.raw(y = stockfuture_core_train$nextday_openchange ,
                        x = as.matrix(stockfuture_core_train[,2:3]),
                        kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.45,NULL))

LL_core_pred <-predict(LL_core_fit,newdata = as.matrix(stockfuture_core_test[,2:3]))


LL_core_pred_train <-predict(LL_core_fit,newdata = as.matrix(stockfuture_core_train[,2:3]))
rmse(actual = stockfuture_core_train$nextday_openchange ,predicted =  LL_core_pred_train)
#樣本內MSE為0.763

LL_core_pred_test <-predict(LL_core_fit,newdata = as.matrix(stockfuture_core_test[,2:3]))
rmse(actual = stockfuture_core_test$nextday_openchange ,predicted =  LL_core_pred_test)
#樣本外MSE為0.493

#See how it fitting
smoothingSpline_LL_core = smooth.spline(stockfuture_core_test$SP500change, LL_core_pred, spar=0.5)

plot(stockfuture_core_test$SP500change , stockfuture_core_test$nextday_openchange)

lines(smoothingSpline_LL_core , col='green' ,lwd = 4)

#####從圖形看配適度#####
{
plot(stockfuture_core_test$SP500change , stockfuture_core_test$nextday_openchange,
     xlab = "SP500價格變化", ylab = "開盤走勢")

lines(smoothingSpline_LL_LASSO , col="red" ,lwd = 4)
lines(smoothingSpline_LL_RF , col="blue" ,lwd = 4)
lines(smoothingSpline_LL_core , col='green' ,lwd = 4)
legend("bottomright",legend = c("LASSO","Random Forest","Both"), lty = 1 , col = c("red","blue","green") ,
       lwd = 4 , cex = 0.7, text.width = 0.4,seg.len = 0.8 ,x.intersp=0.5)
}

