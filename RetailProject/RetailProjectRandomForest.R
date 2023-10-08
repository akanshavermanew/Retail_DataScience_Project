# Building Retail project using random forest model for classification
#here auc score is 0.814

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)
library(vip)
library(rpart.plot)
library(DALEXtra)

setwd("F:/R study materials/Projects/RetailProject")
## ----
st_train=read.csv("store_train.csv",stringsAsFactors = FALSE)
st_test=read.csv("store_test.csv",stringsAsFactors = FALSE)

glimpse(st_train)

vis_dat(st_train)

unique(table(st_train$store_Type))
table(st_train$store_Type)
glimpse(st_train)

st_train$store=as.factor((as.integer(as.logical((st_train$store)))))

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

#building random forest
rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 5)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),     #for random forest passing some values is compulsory otherwise it will give error in decision tree the case was not like this
                       min_n(c(2,10)),levels = 3)

# c(5,25)  means start with 5 and go till 25
# mtry values should be <= features in your table
my_res=tune_grid(
  rf_model,
  store~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

#here you got auc score as 0.814

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(store~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)

#write csv
write.table(test_pred,"Akansha_Verma_P2_part2.csv",row.names=F,col.names = "store")
