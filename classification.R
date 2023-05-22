library(readxl)
data <- read_excel("C:/Users/Nader/Desktop/EBDS/drug_consumption/data.xlsx")
data = data[-11]

# Encoding the target feature as factor
#data[,c(9,10,13:32)] = factor(data[,c(9,10,13:32)], levels = c(0, 1))
data[,c(9,10,12:31)] = lapply(data[,c(9,10,12:31)] , factor)
library(caret)
#Splitting the dataset
set.seed(42) # pour garantir que le meme echantillion est selectionné chaque fois qu'on cree une partition train et test 
train_id <- createDataPartition(data$Cannabis, p=0.80) # train=80%,test=20%
data_train=data[train_id$Resample1,]          #Base Train
data_test=data[-train_id$Resample1,] 

library(dplyr)

data_train_scaled = data_train %>%mutate_if(is.numeric, scale)
data_test_scaled = data_test %>%mutate_if(is.numeric, scale)






#SVM

library(kernlab)
classifier = ksvm(Cannabis ~ .,
                 data = data_train_scaled,
                 type = 'C-svc',
                 kernel = 'polydot',
                 cross = 10
)
print(classifier)
y_pred_svm = predict(classifier, newdata = data_test_scaled[-10])
y_pred_svm

cm = table(data_test_scaled[[10]], y_pred_svm)
cm
confusionMatrix(cm)
accuracy = accuracy= (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
accuracy
plot(classifier, data = data_train_scaled, slice = list(data_train_scaled$Cannabis = 0, data_train_scaled$Cannabis = 1))
library(e1071)
plot_svm_jk(data_train_scaled,svm_model=classifier,surface_plot = T)
#Visualising the Training set results
library(ElemStatLearn)
set = data_train_scaled
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Cannabis', 'individus')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

##################################################################################################

#Random forest

library(randomForest)
set.seed(1234)
rf <-randomForest(Cannabis~.,data=data_train_scaled, ntree=500, mtry=4,importance=TRUE) 
print(rf)
varImpPlot(rf)
#the number of variables tried at each split
floor(sqrt(ncol(data_train_scaled) - 1))
#The number of variables selected at each split
mtry <- tuneRF(data_train_scaled[-10],data_train_scaled$Cannabis, ntreeTry=500,
               stepFactor=1.5,improve=0.01, trace=TRUE, plot=TRUE)
#Find the optimal mtry value
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)
#In this case, mtry = 4 is the best mtry as it has least OOB error.

######################################################################################################
#RandomForest cross validation
library(caret)
trControl <- trainControl(method = "cv",number = 10,search = "grid")

set.seed(1234)
# Run the model
rf_default <- train(Cannabis~.,data = data_train,method = "rf",metric = "Accuracy",trControl = trControl)
# Print the results
print(rf_default)


set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- train(Cannabis~.,
                 data = data_train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 300)
print(rf_mtry)
rf_mtry$bestTune$mtry
max(rf_mtry$results$Accuracy)
best_mtry <- rf_mtry$bestTune$mtry 
best_mtry


store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(Cannabis~.,
                      data = data_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)


store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(Cannabis~.,
                       data = data_train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 24,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)


fit_rf <- train(Cannabis~.,
                data_train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 500,
                maxnodes = 15,
                mytry = 10)
library(partykit)
cforest(Cannabis ~ ., data=data_train)
y_pred_rf <-predict(fit_rf, data_test)
confusionMatrix(y_pred_rf, data_test$Cannabis)

library(ggplot2)
library("party")
x <- ctree(Cannabis ~ ., data=data_train_scaled)

plot(x, type="simple")

#getTree(fit_rf$finalModel,, labelVar=TRUE)


varImp(fit_rf)
ggplot(varImp(fit_rf))
####################################################################################
#Naive bayes
set.seed(1234)
# naive Bayes with cross validation
library(e1071)
trControl <- trainControl(method = "cv",number = 10,search = "grid")
naive_bayes_model <- train(data_train[-10],data_train$Cannabis,method = "nb",metric = "Accuracy",trControl = trControl)
#model = train(train_data,,'nb',trControl=trainControl(method='cv',number=10))
naive_bayes_model
y_pred_naive_model = predict(naive_bayes_model$finalModel,data_test[-10])
cm_bayes_model <- table(data_test$Cannabis, y_pred_naive_model$class)
confusionMatrix(cm_bayes_model)
####################################################################################
#LDA with cross validation
lda_model = train(Cannabis~.,data = data_train,method = "lda",metric = "Accuracy",trControl = trControl)
y_pred_lda = predict(lda_model, data_test[-10])
cm_lda_model <- table(data_test$Cannabis, y_pred_lda)
confusionMatrix(cm_lda_model)
####################################################################################
##########################################################################
#Roc curve
library(pROC)
rocs <- list()
class(y_pred_naive_model$class)
class(y_pred_naive_model)
rocs[["SVM"]] <- roc(data_test$Cannabis, as.numeric(y_pred_svm))
rocs[["Random Forest"]] <- roc(data_test$Cannabis, as.numeric(y_pred_rf))
rocs[["Naive Bayes"]] <- roc(data_test$Cannabis, as.numeric(y_pred_naive_model$class))
rocs[["LDA"]] <- roc(data_test$Cannabis, as.numeric(y_pred_lda))

ggroc(rocs)+
theme_bw() + 
  theme(#legend.title = element_blank(), 
        legend.position = c(0.8, 0.2),
        legend.text = element_text(size=12),
        text = element_text(size=12)) + 
  scale_color_manual(values=c("#00FF00", "#FF0000", "#CCEEFF", "#0000FF"),
    labels=c(paste("SVM: ", round(rocs[["SVM"]]$auc,4), sep=""), 
                              paste("random Forest: ", round(rocs[["Random Forest"]]$auc,4), sep=""), 
                              paste("Naive Bayes: ", round(rocs[["Naive Bayes"]]$auc,4), spe=""),
                              paste("LDA: ", round(rocs[["LDA"]]$auc,4), sep="")
                     )
  )+ labs(color='Area Under the Curve (AUC)') 

rocs[["SVM"]]$auc
rocs[["Random Forest"]]$auc
rocs[["Naive Bayes"]]$auc
rocs[["LDA"]]$auc



