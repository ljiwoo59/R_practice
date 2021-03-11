library(data.table)
library(ggplot2)
library(Rtsne)
library(ClusterR)
library(dummies)

rm(list = ls())

data = fread("/Users/JiWooLee/Desktop/NewSpecies/volume/data/raw/Gene_data.csv")
test = fread("/Users/JiWooLee/Desktop/NewSpecies/volume/data/raw/example_sub.csv")

data[, locus_1 := as.factor(locus_1)]
data[, locus_2 := as.factor(locus_2)]
data[, locus_3 := as.factor(locus_3)]
data[, locus_4 := as.factor(locus_4)]
data[, locus_5 := as.factor(locus_5)]
data[, locus_6 := as.factor(locus_6)]
data[, locus_7 := as.factor(locus_7)]
data[, locus_8 := as.factor(locus_8)]
data[, locus_9 := as.factor(locus_9)]
data[, locus_10 := as.factor(locus_10)]
data[, locus_11 := as.factor(locus_11)]
data[, locus_12 := as.factor(locus_12)]
data[, locus_13 := as.factor(locus_13)]
data[, locus_14 := as.factor(locus_14)]
data[, locus_15 := as.factor(locus_15)]

summary(data)

sample <- data$id
data$id = NULL
d_data<-dummy.data.frame(data)










j_data <- data.frame(lapply(d_data, jitter, factor = 0.01))
pca <- prcomp(j_data)
pca_dt <- data.table(unclass(pca)$x)

tsne_pca <- Rtsne(pca_dt, pca = F)
tsne_dt_pca <- data.table(tsne_pca$Y)
ggplot(tsne_dt_pca, aes(x = V1, y = V2)) + geom_point()


tsne_data <- Rtsne(j_data, pca = T)
tsne_dt_data <- data.table(tsne_data$Y)
ggplot(tsne_dt_data, aes(x = V1, y = V2)) + geom_point()






gmm <- GMM(tsne_dt_data, 3)
saveRDS(gmm, file = "/Users/JiWooLee/Desktop/NewSpecies/volume/model/models")
cluster_prob_data = predict_GMM(tsne_dt_data, 
                                gmm$centroids, 
                                gmm$covariance_matrices, 
                                gmm$weights)$cluster_proba
cluster_prob_data = as.data.table(cluster_prob_data)
cluster_prob_data
ggplot(tsne_dt_data, 
       aes(x = V1, y = V2, col = cluster_prob_data$V2)) +
  geom_point()

cluster = rep("", 2250)
for (i in 1:2250) {
  cluster[i] = names(which.max(cluster_prob_data[i]))
}
ggplot(tsne_dt_data, 
       aes(x = V1, y = V2, col = cluster)) +
  geom_point()

solution = data.table(Id = sample,
                      species1 = cluster_prob_data$V2,
                      species2 = cluster_prob_data$V3,
                      species3 = cluster_prob_data$V1)

fwrite(solution, file = "/Users/JiWooLee/Desktop/NewSpecies/volume/data/processed/mysolution.csv")

