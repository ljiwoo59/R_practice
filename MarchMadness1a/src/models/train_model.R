library(caret)
library(data.table)
library(Metrics)




teams = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MTeams.csv")
teamconference = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MTeamConferences.csv")
teamcoaches = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MTeamCoaches.csv")
regularseason = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv")
nca = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv")
massey = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MMasseyOrdinals.csv")
gamecities = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/MGameCities.csv")
conference = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/Conferences.csv")
cities = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MDataFiles_Stage1/Cities.csv")
test = fread("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MSampleSubmissionStage1_2020.csv")

test
strsplit(test$ID, "_")
unlist(strsplit(test$ID, "_"))


test <- data.table(matrix(unlist(strsplit(test$ID, "_")), ncol = 3, byrow = TRUE))
test
setnames(test, c("V1", "V2", "V3"), c("Season","team_1", "team_2"))
test
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
master$Season <- as.character(master$Season)
pom_ranks$Season <- as.character(pom_ranks$Season)

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

test

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

fwrite(test, file.path("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/interim/test.csv"))
fwrite(train, file.path("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/interim/train.csv"))

train_y <- train$result

glm_model <- glm(result ~ ., family = binomial, data = train)
summary(glm_model)
saveRDS(glm_model, file = file.path("/Users/JiWooLee/Desktop/MarchMadness1a/volume/models/POM_glm.model"))

pred <- predict(glm_model, newdata = test, type = "response")
test$result <- pred
test

sub <- fread(file.path("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/raw/MSampleSubmissionStage1_2020.csv"))
sub$order <- 1:nrow(sub)
sub
teams
teams <- data.table(matrix(unlist(strsplit(sub$ID,"_")), ncol = 3, byrow = TRUE))
setnames(teams, c("V1", "V2", "V3"), c("Season","team_1", "team_2"))
teams

sub$team_1 <- teams$team_1
sub$team_2 <- teams$team_2
sub$Season <- teams$Season


sub$team_1 <- as.character(sub$team_1)
sub$team_2 <- as.character(sub$team_2)
sub$Season <- as.character(sub$Season)
test$team_1 <- as.character(test$team_1)
test$team_2 <- as.character(test$team_2)
test$Season <- teams$Season
test$Season <- as.character(test$Season)

sub$Result <- 1
sub$Result <- NULL
sub
submit <- merge(sub, test, all.x = TRUE, by = c("team_1", "team_2", "Season"))
submit
submit <- submit[order(order)]

submit <- submit[, .(ID, result)]
setnames(submit, "result", "Pred")
submit

fwrite(submit, file.path("/Users/JiWooLee/Desktop/MarchMadness1a/volume/data/processed/submit_POM.csv"))
