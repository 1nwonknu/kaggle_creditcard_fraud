library(xgboost)
library(data.table)
library(magrittr)
library(smotefamily)

setwd("D:/Downloads/creditcard")

# define a custom loss function to train
# the model based on precision
eval_precision <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  xgb_pred_col <- ifelse(preds > 0.5, 1, 0)

  res <- data.table(prediction = xgb_pred_col, actual = labels)
  
  true_positive <- res[prediction == actual & prediction == 1, .N]
  
  false_positive <- res[prediction == 1 & actual == 0, .N]
  
  precision <- true_positive / (true_positive + false_positive)
  
  return(list(metric = "precision", value = precision))
}

dt <- fread("creditcard.csv")

# 40% of original data is used for testing
n_val <- round(nrow(dt) * 0.4)
val <- sample(1:nrow(dt), n_val, replace = FALSE)
val_dt <- dt[val, ]
val_features <- val_dt
val_lable <- val_dt[, Class] %>% as.vector
val_features$Class <- NULL
val_features <- val_features %>% as.matrix

# generate the training data based on
# 60% of the original data
dt_val <- dt[-val, ]
labels <- dt_val[, Class]
features <- copy(dt_val)
features[, Class := NULL]

# oversample the minority class,
# such that the number of fraud/no fraud cases
# are equal in the training set.
out <- SMOTE(X = features, target = labels)

n_train <- round(nrow(out$data) * 1)
train <- sample(1:nrow(out$data), n_train, replace = FALSE)
train_dt <- out$data[train, ]
train_features <- train_dt
train_lable <- train_dt[, class] %>% as.vector
train_features$class <- NULL 
train_features <- train_features %>% as.matrix

# generate the dense matrices for training / testing
xgb.train <- xgb.DMatrix(data = train_features, label = train_lable)
xgb.test <- xgb.DMatrix(data = val_features, label = val_lable)

num_class <- 2
 
params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 10,
  gamma = 0.3,
  subsample = 0.6,
  min_child_weight = 5,
  alpha = 0.5,
  lambda = 10,
  colsample_bytree = 0.5,
  scale_pos_weight = 1,
  objective = "binary:logistic",
  eval_metric = eval_precision
)

xgb.fit <- xgb.train(
  params = params,
  maximize = TRUE,
  data=xgb.train,
  nthreads = 1,
  nrounds = 2000,
  early_stopping_rounds = 1000,
  watchlist = list(train=xgb.train, test=xgb.test),
  verbose = 1
)
# evalue the model
xgb.pred <- predict(xgb.fit, val_features)
xgb_pred <- data.table::as.data.table(xgb.pred)

xgb_pred_col <- xgb_pred[, ifelse(xgb.pred > 0.5 , 1, 0)]

res <- data.table(prediction = xgb_pred_col, actual = val_lable)

accuracy <- res[prediction == actual, .N] / nrow(res)

true_positive <- res[prediction == actual & prediction == 1, .N]
true_negative <- res[prediction == actual & prediction == 0, .N]
false_negative <- res[prediction == 0 & actual == 1, .N]
false_positive <- res[prediction == 1 & actual == 0, .N]

precision <- true_positive / (true_positive + false_positive)

recall <- true_positive / (true_positive + false_negative)

cat(paste0("\n Accuracy: ", accuracy, "\n Precision: ", precision, " \n Recall:", recall))