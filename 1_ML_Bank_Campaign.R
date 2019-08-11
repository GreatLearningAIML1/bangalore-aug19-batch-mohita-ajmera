library(caret)
library(RtutoR)
library(glmnet)
library(dplyr)




# Read data
bank = read.csv("bank.csv")
# bank = bank %>% select(-duration)
gen_exploratory_report_app(bank)


# Data partition
set.seed(100)
#Spliting data as training and test set. Using createDataPartition() function from caret
idx <- createDataPartition(y = bank$y,p = 0.75,list = FALSE)
bank_train <- bank[idx,]
bank_test <- bank[-idx,]


# Dealing with categorical predictors

x <- model.matrix(y ~ ., data = bank)

# Question 1: What does the matrix x contain? How many features does X have?
# Q2: How many features are created for the feaure "marital"?

colnames(x)[grep("marital",colnames(x))]
bank$marital <- factor(bank$marital, levels = c("single","married","divorced"))

# Discussion: How do we interpret coefficients for numeric and factor 
#... variables in logistic regresison?

x_train <- x[idx,]
x_test <- x[-idx,]


# train model
lasso <- glmnet(x_train,bank_train$y,alpha=1,family="binomial", nlambda = 100)
plot(lasso, xvar='lambda')   
print(lasso)


# Finding coefficient with non zero values for different values of lambda
sel_coef = as.matrix(coef(lasso,s=0.1155000))
sel_coef_df = data.frame(Feature = rownames(sel_coef),
                      Coef = sel_coef[,1])
sel_coef_df %>% filter(Coef != 0)


# In class assignment
# What value of lambda results in exactly 5 features?
# What are these features?


# predict
lambda_sel = 0.02
lasso_pred_test = predict(lasso, newx = x_test, type = "class", s = lambda_sel)[,1]
lasso_pred_train = predict(lasso, newx = x_train, type = "class", s = lambda_sel)[,1]
pred_df_test = data.frame(Actual = bank_test$y, Pred = lasso_pred_test)
pred_df_train = data.frame(Actual = bank_train$y, Pred = lasso_pred_train)

# Compute accuracy metrics 
pred_df_test = data.frame(Actual = bank_test$y, Pred = "no")
conf_mat <- confusionMatrix(pred_df_test$Pred, pred_df_test$Actual)
conf_mat$table


# In class assignment
# What is the accuracy if model predicts all cases as "no"?


# Predicting probabilities vs class
lasso_pred_raw_test = predict(lasso, newx = x_test, type = "response", s = lambda_sel)[,1]
lasso_pred_raw_train = predict(lasso, newx = x_train, type = "response", s = lambda_sel)[,1]
pred_df_test = data.frame(Actual = bank_test$y, Pred = lasso_pred_raw_test)
pred_df_train = data.frame(Actual = bank_train$y, Pred = lasso_pred_raw_train)

pROC::auc(pred_df_test$Actual, pred_df_test$Pred)
pROC::auc(pred_df_train$Actual, pred_df_train$Pred)


chk = glm(x_train,bank_train$y, family = "binomial")

# In class assignment
# What is the train and test auc when s = 0.04

# Lets convert the above code to a function
get_auc <- function(lambda_sel) {
  
  lasso_pred_raw_test = predict(lasso, newx = x_test, type = "response", s = lambda_sel)[,1]
  lasso_pred_raw_train = predict(lasso, newx = x_train, type = "response", s = lambda_sel)[,1]
  pred_df_test = data.frame(Actual = bank_test$y, Pred = lasso_pred_raw_test)
  pred_df_train = data.frame(Actual = bank_train$y, Pred = lasso_pred_raw_train)
  
  test_auc = pROC::auc(pred_df_test$Actual, pred_df_test$Pred)
  train_auc = pROC::auc(pred_df_train$Actual, pred_df_train$Pred)
  
  return (list(train_auc = train_auc, test_auc = test_auc))
  
}

# Analyze bias variance trade-off by changing values of lambda
get_auc(0.003)



# gain chart & lift index
pred_df_test$Actual_num <- ifelse(pred_df_test$Actual == "yes",1,0)
gains::gains(pred_df_test$Actual_num, pred_df_test$Pred)


# Using cross validation to identify optimal lambda
cv_lasso = cv.glmnet(x_train, bank_train$y, 
                     family = "binomial", nfold = 5, 
                     type.measure = "auc")

max(cv_lasso$cvm)
max_aux_lambda = which(cv_lasso$cvm == max(cv_lasso$cvm))
cv_lasso$lambda[max_aux_lambda]
cv_lasso$lambda.min









