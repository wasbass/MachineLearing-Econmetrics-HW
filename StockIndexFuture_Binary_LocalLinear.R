#####Library#####
{
  library(np)
  library(readxl)
  library(locfit)
  library(dplyr)
  library(PLRModels)
  library(caret)
  library(ROCR)
}
#####Setting#####
setwd("C:/RRR/")
stockfuture <- read_excel("完整資料.xlsx")
stockfuture$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)

#####PCA#####
PCA <- read.csv("PCA.csv")
stockfuture_PCA <- cbind( positive_revenue =  stockfuture$positive_revenue , PCA)
stockfuture_PCA$positive_revenue <- ifelse(stockfuture$nextday_openchange >= 0,1,0)

stockfuture_PCA_train <- stockfuture_PCA[1:351,]
stockfuture_PCA_test  <- stockfuture_PCA[352:475,]

#Tuning
# alpha_Binary_PCA.cv <-gcvplot(positive_revenue ~
#                                   PC1+PC2+PC3+PC4+PC5+PC6,
#                                 family = "bonomial" ,kern = "gauss",
#                                 maxk = 10000 ,deg = 1 ,stockfuture_PCA_train,
#                                 alpha = seq(0.3,0.95,by=0.05) , df = 2)
# plot(alpha_Binary_PCA.cv)

LL_Binary_PCA_fit <- locfit.raw(y = stockfuture_PCA_train$positive_revenue ,
                                  x = as.matrix(stockfuture_PCA_train[,2:7]),
                                  family = "bonomial",kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.9,NULL))

LL_Binary_PCA_pred_train_raw <-predict(LL_Binary_PCA_fit,newdata = as.matrix(stockfuture_PCA_train[,2:7]))
LL_Binary_PCA_pred_test_raw <-predict(LL_Binary_PCA_fit,newdata = as.matrix(stockfuture_PCA_test[,2:7]))

pred_Binary_PCA_train <- prediction(LL_Binary_PCA_pred_train_raw , stockfuture_PCA_train$positive_revenue) 
perf_Binary_PCA_train <- performance (pred_Binary_PCA_train , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_PCA_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.781

pred_Binary_PCA <- prediction(LL_Binary_PCA_pred_test_raw , stockfuture_PCA_test$positive_revenue) 
perf_Binary_PCA <- performance (pred_Binary_PCA , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_PCA, measure = "auc")@y.values[[1]]
#PCA樣本外AUC為0.708

#####LASSO#####
stockfuture %>%
  dplyr::select(positive_revenue,
                SP500change,
                SP500change_lag2,
                VIXvolume,
                USDchange,
                VIXchange,
                bitchange) -> stockfuture_LASSO

stockfuture_LASSO_train <- stockfuture_LASSO[1:351,]
stockfuture_LASSO_test  <- stockfuture_LASSO[352:475,]

#Tuning
# alpha_Binary_LASSO.cv <-gcvplot(positive_revenue ~
#                                   SP500change+SP500change_lag2+VIXvolume+USDchange+VIXchange+bitchange,
#                                 family = "bonomial" ,kern = "gauss",
#                                 maxk = 10000 ,deg = 1 ,stockfuture_LASSO_train,
#                                 alpha = seq(0.3,0.95,by=0.05) , df = 2)
# plot(alpha_Binary_LASSO.cv)
#看起來alpha取0.9會最好

LL_Binary_LASSO_fit <- locfit.raw(y = stockfuture_LASSO_train$positive_revenue ,
                           x = as.matrix(stockfuture_LASSO_train[,2:7]),
                           family = "bonomial",kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.9,NULL))

LL_Binary_LASSO_pred_train_raw <-predict(LL_Binary_LASSO_fit,newdata = as.matrix(stockfuture_LASSO_train[,2:7]))
LL_Binary_LASSO_pred_test_raw <-predict(LL_Binary_LASSO_fit,newdata = as.matrix(stockfuture_LASSO_test[,2:7]))

pred_Binary_LASSO_train <- prediction(LL_Binary_LASSO_pred_train_raw , stockfuture_LASSO_train$positive_revenue) 
perf_Binary_LASSO_train <- performance (pred_Binary_LASSO_train , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_LASSO_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.840

pred_Binary_LASSO <- prediction(LL_Binary_LASSO_pred_test_raw , stockfuture_LASSO_test$positive_revenue) 
perf_Binary_LASSO <- performance (pred_Binary_LASSO , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_LASSO, measure = "auc")@y.values[[1]]
#LASSO樣本外AUC為0.861 

#####RF#####
stockfuture %>%
  select(positive_revenue,VIXchange,SP500change,SP500change_lag1,VIXvolume,ETHchange,openchange_lag1) -> stockfuture_RF

stockfuture_RF_train <- stockfuture_RF[1:351,]
stockfuture_RF_test  <- stockfuture_RF[352:475,]

#Tuning
# alpha_Binary_RF.cv <-gcvplot(positive_revenue ~
#                                 VIXchange+SP500change+SP500change_lag1+VIXvolume+ETHchange+openchange_lag1,
#                                 family = "bonomial" ,kern = "gauss",
#                                 maxk = 10000 ,deg = 1 ,stockfuture_RF_train,
#                                 alpha = seq(0.3,0.95,by=0.05) , df = 2)
# plot(alpha_Binary_RF.cv)
#看起來alpha取0.9會最好

LL_Binary_RF_fit <- locfit.raw(y = stockfuture_RF_train$positive_revenue ,
                                  x = as.matrix(stockfuture_RF_train[,2:7]),
                                  family = "bonomial",kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.9,NULL))

LL_Binary_RF_pred_train_raw <-predict(LL_Binary_RF_fit,newdata = as.matrix(stockfuture_RF_train[,2:7]))
LL_Binary_RF_pred_test_raw <-predict(LL_Binary_RF_fit,newdata = as.matrix(stockfuture_RF_test[,2:7]))

pred_Binary_RF_train <- prediction(LL_Binary_RF_pred_train_raw , stockfuture_RF_train$positive_revenue) 
perf_Binary_RF_train <- performance (pred_Binary_RF_train , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_RF_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.831

pred_Binary_RF <- prediction(LL_Binary_RF_pred_test_raw , stockfuture_RF_test$positive_revenue) 
perf_Binary_RF <- performance (pred_Binary_RF , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_RF, measure = "auc")@y.values[[1]]
#RF樣本外AUC為0.843 

#####Core#####
stockfuture %>%
  select(positive_revenue,VIXchange,SP500change) -> stockfuture_core

stockfuture_core_train <- stockfuture_core[1:351,]
stockfuture_core_test  <- stockfuture_core[352:475,]

#Tuning
# alpha_Binary_core.cv <-gcvplot(positive_revenue ~
#                                VIXchange+SP500change,
#                              family = "bonomial" ,kern = "gauss",
#                              maxk = 10000 ,deg = 1 ,stockfuture_core_train,
#                              alpha = seq(0.3,0.95,by=0.05) , df = 2)
# plot(alpha_Binary_core.cv)
#看起來alpha取0.4會最好

LL_Binary_core_fit <- locfit.raw(y = stockfuture_core_train$positive_revenue ,
                               x = as.matrix(stockfuture_core_train[,2:3]),
                               family = "bonomial",kern = "gauss",maxk = 10000 ,deg = 1 , alpha = c(0.4,NULL))

LL_Binary_core_pred_train_raw <-predict(LL_Binary_core_fit,newdata = as.matrix(stockfuture_core_train[,2:3]))
LL_Binary_core_pred_test_raw <-predict(LL_Binary_core_fit,newdata = as.matrix(stockfuture_core_test[,2:3]))

pred_Binary_core_train <- prediction(LL_Binary_core_pred_train_raw , stockfuture_core_train$positive_revenue) 
pecore_Binary_core_train <- performance(pred_Binary_core_train , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_core_train, measure = "auc")@y.values[[1]]
#樣本內AUC為0.832

pred_Binary_core <- prediction(LL_Binary_core_pred_test_raw , stockfuture_core_test$positive_revenue) 
pecore_Binary_core <- performance (pred_Binary_core , measure = "tpr" , x.measure = "fpr")
performance(pred_Binary_core, measure = "auc")@y.values[[1]]
#core樣本外AUC為0.857 
