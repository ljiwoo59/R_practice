rm(list = ls())

library(data.table)
library(Rtsne)
library(xgboost)

train_data = fread("/Users/JiWooLee/Desktop/Final/volume/data/raw/training_data.csv")
test_data = fread("/Users/JiWooLee/Desktop/Final/volume/data/raw/test_data.csv")
train_emb = fread("/Users/JiWooLee/Desktop/Final/volume/data/raw/training_emb.csv")
test_emb = fread("/Users/JiWooLee/Desktop/Final/volume/data/raw/test_emb.csv")
example = fread("/Users/JiWooLee/Desktop/Final/volume/data/raw/example_sub.csv")

train_data$code = NA
test_data$code =NA

train_data$code[train_data$subredditcars == 1] <- 0
train_data$code[train_data$subredditCooking == 1] <- 1
train_data$code[train_data$subredditMachineLearning == 1] <- 2
train_data$code[train_data$subredditmagicTCG == 1] <- 3
train_data$code[train_data$subredditpolitics == 1] <- 4
train_data$code[train_data$subredditReal_Estate == 1] <- 5
train_data$code[train_data$subredditscience == 1] <- 6
train_data$code[train_data$subredditStockMarket == 1] <- 7
train_data$code[train_data$subreddittravel == 1] <- 8
train_data$code[train_data$subredditvideogames == 1] <- 9

table(train_data$code)

data = cbind(rbind(train_data[, .(id, text, code)],
                   test_data[, .(id, text, code)]),
             data.frame(lapply(rbind(train_emb,
                                    test_emb),
                              jitter,
                              factor = 0.0001)))

pca = prcomp(data[, 4:515])

pca_coords <- data.table(unclass(pca)$x)

tsne = Rtsne(pca_coords,
             dims = 2,
             perplexity = 100,
             pca = FALSE)

tsne_coords <- data.table(tsne$Y)

idx = which(data$code < 10)

dtrain <- xgb.DMatrix(as.matrix(tsne_coords[idx]),
                      label = data$code[idx],
                      missing = NA)
dtest <- xgb.DMatrix(as.matrix(tsne_coords[-idx]),
                     missing = NA)



# param tunning
# max_depth
{
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 15,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time1 = proc.time()[3]
  XGBm.cv.1 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 2000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time1 = proc.time()[3] - time1
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 10,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time2 = proc.time()[3]
  XGBm.cv.2 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 2000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time2 = proc.time()[3] - time2
  
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 5,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time3 = proc.time()[3]
  XGBm.cv.3 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 16000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time3 = proc.time()[3] - time3
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 1,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  
  time4 = proc.time()[3]
  XGBm.cv.4 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 50000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time4 = proc.time()[3] - time4
  
  

  cat("depth = 15 time = ", time1, 
      "mlogloss = ", min(XGBm.cv.1$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.1$evaluation_log$test_mlogloss_mean), "\n")
  cat("depth = 10 time = ", time2, 
      "mlogloss = ", min(XGBm.cv.2$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.2$evaluation_log$test_mlogloss_mean), "\n")
  cat("depth =  5 time = ", time3, 
      "mlogloss = ", min(XGBm.cv.3$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.3$evaluation_log$test_rmse_mean), "\n")
  cat("depth =  1 time = ", time4, 
      "mlogloss = ", min(XGBm.cv.4$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.4$evaluation_log$test_mlogloss_mean), "\n")
  
  save(time1, time2, time3, time4, 
       XGBm.cv.1, XGBm.cv.2, XGBm.cv.3, XGBm.cv.4, 
       file = "/Users/JiWooLee/Desktop/Final/volume/model/XBGm.cv.1-4.rdata")
}

# max depth
{
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 7,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time5 = proc.time()[3]
  XGBm.cv.5 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 5000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time5 = proc.time()[3] - time5
  
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 6,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time6 = proc.time()[3]
  XGBm.cv.6 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 5000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time6 = proc.time()[3] - time6
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 5,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time7 = proc.time()[3]
  XGBm.cv.7 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 5000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time7 = proc.time()[3] - time7
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 4,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time8 = proc.time()[3]
  XGBm.cv.8 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 10000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time8 = proc.time()[3] - time8
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 3,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  time9 = proc.time()[3]
  XGBm.cv.9 <- xgb.cv(params        = param,
                      nfold         = 5,
                      nrounds       = 10000,  
                      missing       = NA,    
                      data          = dtrain, 
                      print_every_n = 10)
  time9 = proc.time()[3] - time9
  
  

  cat("depth = 7 time = ", time5, 
      "mlogloss = ", min(XGBm.cv.5$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.5$evaluation_log$test_mlogloss_mean), "\n")
  cat("depth = 6 time = ", time6, 
      "mlogloss = ", min(XGBm.cv.6$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.6$evaluation_log$test_mlogloss_mean), "\n")
  cat("depth = 5 time = ", time7, 
      "mlogloss = ", min(XGBm.cv.7$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.7$evaluation_log$test_mlogloss_mean), "\n")
  cat("depth = 4 time = ", time8, 
      "mlogloss = ", min(XGBm.cv.8$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.8$evaluation_log$test_mlogloss_mean), "\n")
  cat("depth = 3 time = ", time9, 
      "mlogloss = ", min(XGBm.cv.9$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.9$evaluation_log$test_mlogloss_mean), "\n")
  
  save(time5, time6, time7, time8, time9, 
       XGBm.cv.5, XGBm.cv.6, XGBm.cv.7, XGBm.cv.8,XGBm.cv.9, 
       file = "/Users/JiWooLee/Desktop/Final/volume/model/XBGm.cv.5-9.rdata")
}

#eta
{
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 4,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  
  timeeta1 = proc.time()[3]
  XGBm.cv.eta1 <- xgb.cv(params        = param,
                         nfold         = 5,
                         nrounds       = 10000,  
                         missing       = NA,    
                         data          = dtrain, 
                         print_every_n = 1000)
  timeeta1 = proc.time()[3] - timeeta1
  
  param$eta = 0.01
  
  timeeta2 = proc.time()[3]
  XGBm.cv.eta2 <- xgb.cv(params        = param,
                         nfold         = 5,
                         nrounds       = 10000,  
                         missing       = NA,    
                         data          = dtrain, 
                         print_every_n = 1000)
  timeeta2 = proc.time()[3] - timeeta2
  
  param$eta = 0.1
  
  timeeta3 = proc.time()[3]
  XGBm.cv.eta3 <- xgb.cv(params        = param,
                         nfold         = 5,
                         nrounds       = 10000,  
                         missing       = NA,    
                         data          = dtrain, 
                         print_every_n = 1000)
  timeeta3 = proc.time()[3] - timeeta3
  
  param$eta = 1
  
  timeeta4 = proc.time()[3]
  XGBm.cv.eta4 <- xgb.cv(params        = param,
                         nfold         = 5,
                         nrounds       = 10000,  
                         missing       = NA,    
                         data          = dtrain, 
                         print_every_n = 1000)
  timeeta4 = proc.time()[3] - timeeta4
  
  # Where is test RMSE minimized.
  
  # load("XBGm.cv.eta.rdata")
  cat("eta = 0.001 time = ", timeeta1, 
      "mlogloss = ", min(XGBm.cv.eta1$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.eta1$evaluation_log$test_rmse_mean), "\n")
  cat("eta = 0.01  time = ", timeeta2, 
      "mlogloss = ", min(XGBm.cv.eta2$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.eta2$evaluation_log$test_rmse_mean), "\n")
  cat("eta = 0.1   time = ", timeeta3, 
      "mlogloss = ", min(XGBm.cv.eta3$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.eta3$evaluation_log$test_rmse_mean), "\n")
  cat("eta = 1     time = ", timeeta4, 
      "mlogloss = ", min(XGBm.cv.eta4$evaluation_log$test_mlogloss_mean), 
      "idx = ", which.min(XGBm.cv.eta4$evaluation_log$test_rmse_mean), "\n")
  save(timeeta1, timeeta2, timeeta3, timeeta4, 
       XGBm.cv.eta1, XGBm.cv.eta2, XGBm.cv.eta3, XGBm.cv.eta4, 
       file = "/Users/JiWooLee/Desktop/Final/volume/model/XBGm.cv.eta.rdata")
}

# subsample&colsample
{
  subsample = c(1.0, 0.95, 0.90, 0.85, 0.80)
  colsample_bytree = c(1.0, 0.95, 0.90, 0.85, 0.80)
  results = data.table(subsample = rep(0, 25),
                       colsample_bytree = rep(0, 25),
                       time             = rep(0, 25),
                       rmse             = rep(0, 25),
                       idx              = rep(0, 25))
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 4,
                eta                 = 0.001,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  i = 1
  for (ss in subsample) {
    for (cs in colsample_bytree) {
      cat("\nss = ", ss, "cs = ", cs, "\n\n")
      param$subsample = ss
      param$colsample_bytree = cs
      time = proc.time()[3]
      XGBm.cv <- xgb.cv(params        = param,
                        nfold         = 5,
                        nrounds       = 10000,  
                        missing       = NA,    
                        data          = dtrain, 
                        print_every_n = 1000)
      results$subsample[i] = ss
      results$colsample_bytree[i] = cs
      results$time[i] = proc.time()[3] - time
      results$rmse[i] = min(XGBm.cv$evaluation_log$test_mlogloss_mean)
      results$idx[i]  = which.min(XGBm.cv$evaluation_log$test_mlogloss_mean)
      print(results[1:i])
      i = i + 1
    }
  }

  print(results)
  save(results, file = "/Users/JiWooLee/Desktop/Final/volume/model/XBGm.cv.subsample.rdata")
}

# gamma&minchildw
{
  gamma = c(0.2, 0.1, 0.05, 0.02, 0.01)
  min_child_weight = c(1, 5, 10)
  results = data.table(gamma            = rep(0, 15),
                       min_child_weight = rep(0, 15),
                       time             = rep(0, 15),
                       mlogloss             = rep(0, 15),
                       idx              = rep(0, 15))
  
  param <- list(objective           = "multi:softprob",
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                tree_method         = 'hist',
                num_class           = 10,
                max_depth           = 4,
                eta                 = 0.001,
                subsample           = 0.90,
                colsample_bytree    = 0.85,
                gamma               = 0.2,
                min_child_weight    = 1
  )
  i = 1
  for (gam in gamma) {
    for (mcw in min_child_weight) {
      cat("\ngamma = ", gam, "min_child_weight = ", mcw, "\n\n")
      param$gamma = gam
      param$min_child_weight = mcw
      time = proc.time()[3]
      XGBm.cv <- xgb.cv(params        = param,
                        nfold         = 5,
                        nrounds       = 20000,  
                        missing       = NA,     
                        data          = dtrain, 
                        print_every_n = 1000)    
      results$gamma[i] = gam                    
      results$min_child_weight[i] = mcw         
      results$time[i] = proc.time()[3] - time   
      results$rmse[i] = min(XGBm.cv$evaluation_log$test_mlogloss_mean)
      results$idx[i]  = which.min(XGBm.cv$evaluation_log$test_mlogloss_mean)
      print(results[1:i])
      i = i + 1
    }
  }
 
  print(results)
  save(results, file = "/Users/JiWooLee/Desktop/Final/volume/model/XBGm.cv.gamma.child.rdata")
}

#param
param <- list(objective           = "multi:softprob",
              booster             = "gbtree",
              eval_metric         = "mlogloss",
              tree_method         = 'hist',
              num_class           = 10,
              max_depth           = 4,
              eta                 = 0.001,
              subsample           = 0.90,
              colsample_bytree    = 0.85,
              gamma               = 0.02,
              min_child_weight    = 5
)

fit = xgb.cv(params = param,
             nfold = 5,
             nrounds = 35000,
             missing = NA,
             data = dtrain,
             print_every_n = 100)
print(proc.time()[3] - time)

nrounds = 8600

watchlist <- list(train = dtrain)

fit <- xgb.train(params = param,
                 nrounds = nrounds,
                 missing = NA,
                 data = dtrain,
                 watchlist = watchlist,
                 print_every_n = 100)
saveRDS(fit, file = "/Users/JiWooLee/Desktop/Final/volume/model/models")

pred = predict(fit, dtest)

M = matrix(pred, ncol = 10, byrow = TRUE)

solution = cbind(test_data$id, as.data.table(M))
names(solution) = names(example)
head(solution)

fwrite(solution, file = "/Users/JiWooLee/Desktop/Final/volume/data/processed/solution.csv")
