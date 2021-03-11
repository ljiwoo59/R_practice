library(data.table)
library(rvest)

train = fread("/Users/JiWooLee/Desktop/HousePriceRegression/HousePriceRegression/volume/data/raw/Stat_380_train.csv")
test = fread("/Users/JiWooLee/Desktop/HousePriceRegression/HousePriceRegression/volume/data/raw/Stat_380_test.csv")

library(caret)
library(data.table)
library(Metrics)
library(DataComputing)

test
train


model <- lm(SalePrice ~LotArea+OverallQual+OverallCond+FullBath+HalfBath+TotRmsAbvGrd+YearBuilt+TotalBsmtSF+BedroomAbvGr+GrLivArea+PoolArea, data = train)
saveRDS(model, file = "models")
summary(model)

p <- predict(model, test)

pred <- as.data.frame(p)

merged <- cbind(test, pred)

submit = test[, .(Id,p)]
colnames(submit)[2] = "SalePrice"

submit
p
fwrite(submit, file = "/Users/JiWooLee/Desktop/HousePriceRegression/HousePriceRegression/volume/data/processed/submit_lm.csv")

