library(caret)
library(data.table)
library(Metrics)

teams = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/Teams.csv")
teamconference = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/TeamConferences.csv")
teamcoaches = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/TeamCoaches.csv")
regularseason = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/RegularSeasonDetailedResults.csv")
nca = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/NCAATourneyDetailedResults.csv")
massey = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/MasseyOrdinals_thru_2019_day_128.csv")
gamecities = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/GameCities.csv")
conference = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/Conferences.csv")
cities = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/Cities.csv")
test = fread("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/examp_sub.csv")

test
strsplit(test$id, "_")
unlist(strsplit(test$id, "_"))


test <- data.table(matrix(unlist(strsplit(test$id, "_")), ncol = 2, byrow = TRUE))
test
setnames(test, c("V1", "V2"), c("team_1", "team_2"))
test
test$Season <- 2019
test$DayNum <- 133
test$result <- 0.5
test

train <- rbind(regularseason, nca)
train
train <- train[,.(WTeamID, LTeamID, Season, DayNum)]
setnames(train,c("WTeamID","LTeamID"),c("team_1","team_2"))
train

train$result <- 1

master <- rbind(train,test)
master$team_1 <- as.character(master$team_1)
master$team_2 <- as.character(master$team_2)

massey

massey$DayNum <- massey$RankingDayNum + 1
massey
pom_ranks <- massey[SystemName == "POM", .(Season, DayNum, TeamID, OrdinalRank)]
pom_ranks
setnames(pom_ranks, "TeamID", "team_1")

pom_ranks
pom_ranks$team_1 <- as.character(pom_ranks$team_1)

setkey(master, Season,team_1, DayNum)
setkey(pom_ranks, Season, team_1, DayNum)

master
pom_ranks

master <- pom_ranks[master, roll = T]
master
setnames(master,"OrdinalRank", "team_1_POM")


setnames(pom_ranks,"team_1","team_2")
setkey(master,Season,team_2,DayNum)
setkey(pom_ranks,Season,team_2,DayNum)

master <- pom_ranks[master,roll = T]

setnames(master,"OrdinalRank","team_2_POM")

master
master[, POM_dif := team_2_POM - team_1_POM]
master
master <- master[order(Season, DayNum)]

master <- master[,.(team_1,team_2, POM_dif, result)]


master <- master[!is.na(master$POM_dif)]

test <- master[result == 0.5]
train <- master[result == 1]

rand_idx <- sample(1:nrow(train), nrow(train) * 0.5)
train_a <- train[rand_idx, ]
train_b <- train[!rand_idx, ]

train_b$result <- 0
train_b$POM_dif <- -1 * train_b$POM_dif

train <- rbind(train_a, train_b)

train
test

fwrite(test, file.path("/Users/JiWooLee/Desktop/MarchMadness/volume/data/interim/test.csv"))
fwrite(train, file.path("/Users/JiWooLee/Desktop/MarchMadness/volume/data/interim/train.csv"))

train_y <- train$result

glm_model <- glm(result ~ ., family = binomial, data = train)
summary(glm_model)
saveRDS(glm_model, file = file.path("/Users/JiWooLee/Desktop/MarchMadness/volume/models/POM_glm.model"))

pred <- predict(glm_model, newdata = test, type = "response")
test$result <- pred
test

sub <- fread(file.path("/Users/JiWooLee/Desktop/MarchMadness/volume/data/raw/examp_sub.csv"))
sub$order <- 1:nrow(sub)
teams <- data.table(matrix(unlist(strsplit(sub$id,"_")), ncol = 2, byrow = TRUE))
setnames(teams, c("V1", "V2"), c("team_1", "team_2"))

sub$team_1 <- teams$team_1
sub$team_2 <- teams$team_2

sub$team_1 <- as.character(sub$team_1)
sub$team_2 <- as.character(sub$team_2)
test$team_1 <- as.character(test$team_1)
test$team_2 <- as.character(test$team_2)

sub$Result <- NULL
submit <- merge(sub, test, all.x = TRUE, by = c("team_1", "team_2"))
submit <- submit[order(order)]

submit <- submit[, .(id, result)]
submit

fwrite(submit, file.path("/Users/JiWooLee/Desktop/MarchMadness/volume/data/processed/submit_POM.csv"))
