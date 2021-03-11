library(data.table)

train = fread("/Users/JiWooLee/Desktop/CoinRegression/volume/data/raw/train_file.csv")
test = fread("/Users/JiWooLee/Desktop/CoinRegression/volume/data/raw/test_file.csv")

fit = glm(result ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10, data = train, family = binomial)
saveRDS(fit, file="/Users/JiWooLee/Desktop/CoinRegression/volume/models/models")
coef(fit)

probs = fit$fitted.values

probs1 = as.data.table(predict(fit, test, type = "response"))

results <- cbind(test, probs1)
submit = results[, .(id,probs1)]

colnames(submit)[2] = "result"

fwrite(submit, file = "/Users/JiWooLee/Desktop/CoinRegression/volume/data/processed/submit_coinr.csv")

submit
