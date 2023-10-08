# Building Retail project using logistic regression model for classification
#can refer another code from here https://rpubs.com/Rajib/530022

#auc score came 0.7 the more the auc score the better the model is

#here we have not find cutoff as it was not asked to submit hard classes(cutoff)

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)

setwd("F:/R study materials/Projects/RetailProject")
## ----
st_train=read.csv("store_train.csv",stringsAsFactors = FALSE)
st_test=read.csv("store_test.csv",stringsAsFactors = FALSE)

glimpse(st_train)

vis_dat(st_train)

unique(table(st_train$store_Type))
table(st_train$store_Type)
glimpse(st_train)


summary(dp_pipe)
dp_pipe=recipe(store ~ .,data=st_train) %>% 
  update_role(Id,new_role = "drop_vars") %>% 
  update_role(countyname,storecode,Areaname,countytownname,state_alpha,store_Type,new_role="to_dummies") %>% 
  
  step_rm(has_role("drop_vars")) %>% 
  #step_novel(Type,Method,SellerG,CouncilArea) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe,new_data = NULL)
test=bake(dp_pipe,new_data=st_test)

vis_dat(train)


set.seed(2)
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,]
t2=train[-s,]

## remove vars with vif higher than 10
for_vif=lm(store~.,data=t1)
summary(for_vif)

sort(vif(for_vif),decreasing = T)[1:3]

for_vif=lm(store~. -store_Type_X__other__ -Areaname_X__other__ -sales0 -sales2
           -sales3 -state_alpha_X__other__ -sales1    ,data=t1)

#building logistic regression model(Minimized the values based on vif values greater than 10)
log_fit=glm(store~.-store_Type_X__other__ -Areaname_X__other__ -sales0 -sales2
            -sales3 -state_alpha_X__other__ -sales1,data=t1,family = "binomial")
summary(log_fit)
sort(vif(log_fit),decreasing = T)[1:3]

log_fit=stats::step(log_fit)

summary(log_fit)
formula(log_fit)

#Minmizing based on p values after taking actual formula it is reduced to this much after removing greater p value columns
log_fit=glm(store ~ sales4 + State + CouSub + population + storecode_X__other__ + 
              state_alpha_GA + state_alpha_MA + state_alpha_ME + state_alpha_NC + 
              state_alpha_NH + state_alpha_TX + state_alpha_VA + state_alpha_VT  ,data=t1,family="binomial")

summary(log_fit)

log_fit=glm(store ~ sales4 + State + CouSub + population + storecode_X__other__ + 
              state_alpha_GA + state_alpha_MA  + state_alpha_NC + 
              state_alpha_NH + state_alpha_TX + state_alpha_VA + state_alpha_VT  ,data=t1,family="binomial")

#predicting auc score
val.score=predict(log_fit,newdata = t2,type='response')
pROC::auc(pROC::roc(t2$store,val.score))

#write csv
write.table(test_pred,"Akansha_Verma_P2_part2.csv",row.names=F,col.names = "store")
