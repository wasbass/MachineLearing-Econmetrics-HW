require(glmnet)
require(ggplot2)
require(tidyverse)

#####Ridge Regression#####

rid_betas = as.data.frame(matrix(data=0,nrow = 201,ncol = 25))
ridoutmse_lambda = as.data.frame(matrix(data=0,nrow = 201,ncol = 1))
lambda = seq(from = 0,to = 2 , by = 0.01)
rid_beta1 = as.data.frame(matrix(data=0,nrow = 500,ncol = 201))

#replicate the simulations 500 times
for(i in 1:500){
  data=data.frame()
  
  for(j in 1:25){
    data[1:50,j] <- rnorm(50)
  }
  
  data$y = 2 + data$V1 - data$V2 + rnorm(50)
  
  #Estimate Y by previous model and calculate MSE
  if(i>1){
    mse_each = c()
    newX <- model.matrix(~.-y,data=data)[,-1]
    predict =predict.glmnet(object = last_model , newx = newX ,  type = "response")
    for(l in 1:201){
      mse_each[l] = mean(
        (( data$y-t(predict.glmnet(object = last_model , newx = newX ,  type = "response")[,202-l]))^2)
      )
    }
    ridoutmse_lambda = ridoutmse_lambda + mse_each
  }
  rid_reg = glmnet( data[,1:25] ,data[,26],nlambda = 25,alpha = 0,family='gaussian',lambda = lambda)
  
  rid_beta1[i,1:201] = t(data.frame(rid_reg$beta[1,201:1]))
  
  #summing up the coefficients
  for(k in 1:25){
    rid_betas[,k] = rid_betas[,k] + as.data.frame(rid_reg$beta[k,201:1])
  }
  
  if(i == 500){
    rid_betas = rid_betas/500    #get the average of coefficients
    ridoutmse_lambda = ridoutmse_lambda/499  #get the average of mse
    break
  }
  last_model = rid_reg
}

rid_outcome = rid_betas
rid_outcome$lambda = lambda
rid_outcome$outmse = ridoutmse_lambda$V1
rid_outcome$beta1var = 0
rid_outcome$beta1bias2 =0

#Getting the Variance of Beta1
for (i in 1:201){
  rid_outcome$beta1var[i] = var(rid_beta1[,i])
  rid_outcome$beta1bias2[i] = (rid_outcome$V1[i]-rid_outcome$V1[1])**2
}

rid_outcome$beta1mse = rid_outcome$beta1var + rid_outcome$beta1bias2

#Plot the convergence rate of the coefficients , the relation between PMSE  , Variance of Beta1 and lambda also.
rid_plotcoef <- rid_outcome[c(1:26)]%>%
  gather(key = "variable" , value = "value",-lambda)

painted <- c("red", rep("black",times = 10) ,"red",rep("black",times = 13))

ggplot(rid_plotcoef , aes(lambda,value)) +
  geom_line(aes(color = variable))+
  scale_color_manual(values = painted)+
  theme_bw(base_family = 'Times')+
  theme(legend.position = 'none')

ggplot(rid_outcome ,aes(lambda,outmse)) + geom_line() +theme_bw(base_family = 'Times')

rid_plotvar <- rid_outcome[c(26,28:30)]%>%
  gather(key = "variable" , value = "var",-lambda)

ggplot(rid_plotvar , aes(lambda,var)) +
  geom_line(aes(color = variable))+
  scale_color_manual(values = c("red","black","green3"))+
  theme_bw(base_family = 'Times')

#####LASSO Regression#####

las_betas = as.data.frame(matrix(data=0,nrow = 201,ncol = 25))
lasoutmse_lambda = as.data.frame(matrix(data=0,nrow = 201,ncol = 1))
lambda = seq(from = 0,to = 2 , by = 0.01)
las_beta1 = as.data.frame(matrix(data=0,nrow = 500,ncol = 201))

#replicate the simulations 500 times
for(i in 1:500){
  data=data.frame()
  
  for(j in 1:25){
    data[1:50,j] <- rnorm(50)
  }
  
  data$y = 2 + data$V1 - data$V2 + rnorm(50)
  
  #Estimate Y by previous model and calculate MSE
  if(i>1){
    mse_each = c()
    newX <- model.matrix(~.-y,data=data)[,-1]
    predict =predict.glmnet(object = last_model , newx = newX ,  type = "response")
    for(l in 1:201){
      mse_each[l] = mean(
        (( data$y-t(predict.glmnet(object = last_model , newx = newX ,  type = "response")[,202-l]))^2)
      )
    }
    lasoutmse_lambda = lasoutmse_lambda + mse_each
  }
  las_reg = glmnet( data[,1:25] ,data[,26],nlambda = 25,alpha = 1,family='gaussian',lambda = lambda)
  
  las_beta1[i,1:201] = t(data.frame(las_reg$beta[1,201:1]))
  
  #summing up the coefficients
  for(k in 1:25){
    las_betas[,k] = las_betas[,k] + as.data.frame(las_reg$beta[k,201:1])
  }
  
  if(i == 500){
    las_betas = las_betas/500    #get the average of coefficients
    lasoutmse_lambda = lasoutmse_lambda/499  #get the average of mse
    break
  }
  last_model = las_reg
}

las_outcome = las_betas
las_outcome$lambda = lambda
las_outcome$outmse = lasoutmse_lambda$V1
las_outcome$beta1var = 0
las_outcome$beta1bias2 =0

#Getting the Variance of Beta1
for (i in 1:201){
  las_outcome$beta1var[i] = var(las_beta1[,i])
  las_outcome$beta1bias2[i] = (las_outcome$V1[i]-las_outcome$V1[1])**2
}

las_outcome$beta1mse = las_outcome$beta1var + las_outcome$beta1bias2

#Plot the convergence rate of the coefficients , the relation between PMSE  , Variance of Beta1 and lambda also.
las_plotcoef <- las_outcome[c(1:26)]%>%
  gather(key = "variable" , value = "value",-lambda)

painted <- c("red", rep("black",times = 10) ,"red",rep("black",times = 13))

ggplot(las_plotcoef , aes(lambda,value)) +
  geom_line(aes(color = variable))+
  scale_color_manual(values = painted)+
  theme_bw(base_family = 'Times')+
  theme(legend.position = 'none')

ggplot(las_outcome ,aes(lambda,outmse)) + geom_line() +theme_bw(base_family = 'Times')

las_plotvar <- las_outcome[c(26,28:30)]%>%
  gather(key = "variable" , value = "var",-lambda)

ggplot(las_plotvar , aes(lambda,var)) +
  geom_line(aes(color = variable))+
  scale_color_manual(values = c("red","black","green3"))+
  theme_bw(base_family = 'Times')