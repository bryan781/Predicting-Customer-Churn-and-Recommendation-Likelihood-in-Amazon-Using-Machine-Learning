library(tidyverse)
library(caret)
library(xgboost)
library(pROC)
library(dplyr)
library(ggplot2)
library(ggcorrplot)

data <- read.csv("C:\\Users\\Bryan\\Documents\\NJU S2\\Semester 1\\Data Analytics\\final project\\amazon_behavior_data_2024.csv")

#EDA
glimpse(data)

summary(data)

anyNA(data)

colnames(data)

bins<-c(0,20,30,40,50,Inf)
labels<-c("15-25","25-35","35-45","45-55","55+")

data$Age_Category<-cut(data$age, breaks=bins, labels = labels, right=FALSE)

categorical_cols <- c("age", "Gender", "Purchase_Categories", "Purchase_Frequency", "Personalized_Recommendation_Frequency", "Recommendation_Helpfulness", "Browsing_Frequency", "Product_Search_Method", "Search_Result_Exploration", "Add_to_Cart_Browsing", "Cart_Completion_Frequency", "Cart_Abandonment_Factors", "Saveforlater_Frequency", "Review_Left", "Review_Reliability","Review_Helpfulness", "Service_Appreciation", "Improvement_Areas")
data[categorical_cols]<-data[categorical_cols]%>%mutate_all(~ as.numeric(as.factor(.)))
#High & Low Correlated Matrixes
correlation_matrix<-cor(data%>%select_if(is.numeric),use="complete.obs")
threshold<-0.5
h_corr<-which(abs(correlation_matrix)>threshold, arr.ind=TRUE)
for(i in 1:nrow(h_corr)){
  feature1<-rownames(correlation_matrix)[h_corr[i,1]]
  feature2<-colnames(correlation_matrix)[h_corr[i,2]]
  
  if(feature1!=feature2){
    print(paste(feature1, "and", feature2, "are high correlated"))
  }
}

correlation_matrix<-cor(data%>%select_if(is.numeric),use="complete.obs")
threshold2<-0.0009
l_corr<-which(abs(correlation_matrix)>threshold2, arr.ind=TRUE)
if(length(l_corr)>0){
  for(i in 1:nrow(l_corr)){
    feature1<-rownames(correlation_matrix)[l_corr[i,1]]
    feature2<-colnames(correlation_matrix)[l_corr[i,2]]
    
    if(feature1!=feature2){
      print(paste(feature1, "and", feature2, "are low correlated"))
    }
    
  }
}

#Heatmap
correlation_matrix<-cor(data%>%select_if(is.numeric), use="complete.obs")
ggcorrplot(correlation_matrix, method="square", lab=TRUE, lab_size=2, tl.cex=5, tl.srt=90, hc.order=TRUE, outline.col="white")

ggplot(data, aes(x=Review_Reliability, fill=as.factor(Review_Left)))+
  geom_bar(position="dodge")+
  scale_fill_manual(values=c("#1f77b4", "#ff7f0e"))+
  labs(x="Review_Reliability", y="Count", fill="Review_Left")+
  theme_minimal()+
  theme(legend.position="top")

#Amazon Users Age Distribution Chart
age_counts<-data%>%count(Age_Category)%>%mutate(Percentage=(n/sum(n))*100)
age_counts$Age_Category<-factor(age_counts$Age_Category, levels=unique(data$Age_Category))

ggplot(age_counts, aes(x= "", y=Percentage, fill=Age_Category)) +
  geom_bar(stat = "identity", width=1) +
  coord_polar("y", start=0) +
  theme_void() +   
  labs(title ="Age distribution of Amazon users", fill="Age Category") +
  geom_text(aes(label = sprintf("%.1f%%", Percentage)), position = position_stack(vjust = 0.5)) +
  theme(legend.position="bottom")

ggplot(data, aes(x = Purchase_Frequency, fill = as.factor(Gender))) +
  geom_bar(position = "dodge") + 
  scale_fill_brewer(palette = "Set2") +  labs(x = "Purchase Frequency", y = "Count", fill = "Gender") + theme_minimal() +   theme(legend.position = "top")

## Machine Learning

#Remove Timestamps
data<-data%>%select(-Timestamp)

#Convert target variable to binary factor
data$Shopping_Satisfaction<-as.factor(ifelse(data$Shopping_Satisfaction>3, "Satisfied", "Not Satisfied"))

#Categorical Variables as factors
data<-data%>%mutate(across(where(is.character), as.factor))

#Train Test Split
set.seed(123)
trainIndex<-createDataPartition(data$Shopping_Satisfaction, p=0.8, list=FALSE)
train_dat<-data[trainIndex, ]
test_dat<-data[-trainIndex, ]

# Correlation filtering 
numeric_cols<-sapply(train_dat, is.numeric)
corr_matrix<-cor(train_dat[, numeric_cols], use="pairwise.complete.obs")
high_corr<-findCorrelation(corr_matrix, cutoff=0.75)

if(length(high_corr)>0){
  train_dat<-train_dat[, -high_corr]
  test_dat<-test_dat[, -high_corr]
}

# One-Hot Encoding
dummy <- dummyVars(Shopping_Satisfaction ~ ., data = train_dat)
train_X <- predict(dummy, train_dat)
test_X <- predict(dummy, test_dat)
train_y <- train_dat$Shopping_Satisfaction
test_y <- test_dat$Shopping_Satisfaction

# Cross-validation for Logistic Regression
train_control <- trainControl(method = "cv", number = 5)
cv_logistic <- train(Shopping_Satisfaction ~ ., data = train_dat,  method = "glm", family = "binomial", trControl = train_control)
print("Logistic Regression - Cross Validation Results:")
print(cv_logistic)

#tuning grid for XGBoost
tune_grid <- expand.grid(
  nrounds = 100, eta = 0.1, max_depth = 6, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1,  subsample = 0.8)

# Cross-validation for XGBoost
cv_xgb <- train(Shopping_Satisfaction ~ ., data = train_dat, method = "xgbTree", trControl = train_control, tuneGrid = tune_grid)
print("XGBoost - Cross Validation Results:")
print(cv_xgb)

## ------------------- LOGISTIC REGRESSION -------------------
logistic_model <- glm(train_y ~ ., data = as.data.frame(train_X) %>% mutate(train_y = train_y), family = binomial)
logistic_preds_prob <- predict(logistic_model, newdata = as.data.frame(test_X), type = "response")
logistic_preds <- ifelse(logistic_preds_prob > 0.5, 1, 0)

# ROC Curve for Logistic Regression
logistic_roc <- roc(as.numeric(test_y) - 1, logistic_preds_prob)
plot(logistic_roc, col = "blue", main = "ROC Curve - Logistic Regression")
auc(logistic_roc)

summary(logistic_model)

## ------------------- XGBOOST -------------------
for (col in names(train_dat)) {
  if (is.factor(train_dat[[col]])) {
    levels <- unique(c(levels(train_dat[[col]]), levels(test_data[[col]])))
    train_dat[[col]] <- factor(train_dat[[col]], levels = levels)
    test_dat[[col]] <- factor(test_dat[[col]], levels = levels)
    train_dat[[col]] <- as.numeric(train_dat[[col]])
    test_dat[[col]] <- as.numeric(test_dat[[col]])
  }
}

# Prepare data for XGBoost
xgb_train <- xgb.DMatrix(data = as.matrix(train_dat %>% select(-Shopping_Satisfaction)), label = as.numeric(train_y) - 1)
xgb_test <- xgb.DMatrix(data = as.matrix(test_dat %>% select(-Shopping_Satisfaction)), label = as.numeric(test_y) - 1)

# XGBoost with hyperparameter tuning
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(params = params, data = xgb_train, nrounds = 100, verbose = 0)
xgb_preds_prob <- predict(xgb_model, xgb_test)
xgb_preds <- ifelse(xgb_preds_prob > 0.5, 1, 0)

# ROC Curve for XGBoost
xgb_roc <- roc(as.numeric(test_y) - 1, xgb_preds_prob)
plot(xgb_roc, col = "red", main = "ROC Curve - XGBoost")
auc(xgb_roc)

#  Confusion Matrices & Accuracy
logistic_cm <- confusionMatrix(logistic_preds_class, test_y)
xgb_cm <- confusionMatrix(xgb_preds_class, test_y)

# Results
print("Logistic Regression - Confusion Matrix:")
print(logistic_cm)

# Metrics for Logistic Regression
cat("\n Logistic Regression - Precision:", logistic_cm$byClass["Precision"], "\n")
cat("Logistic Regression - Recall:", logistic_cm$byClass["Recall"], "\n")

#F1 Score Logistic Regression
logistic_f1 <- 2 * ((logistic_cm$byClass["Precision"] * logistic_cm$byClass["Recall"]) /
                      (logistic_cm$byClass["Precision"] + logistic_cm$byClass["Recall"]))

#XGBoost Confusion Matrix
print("XGBoost - Confusion Matrix:")
print(xgb_cm)

# Print additional metrics for XGBoost
cat("\n XGBoost - Precision:", xgb_cm$byClass["Precision"], "\n")
cat("XGBoost - Recall:", xgb_cm$byClass["Recall"], "\n")

#F1 score for XGBoost
xgb_f1 <- 2 * ((xgb_cm$byClass["Precision"] * xgb_cm$byClass["Recall"]) /
                 (xgb_cm$byClass["Precision"] + xgb_cm$byClass["Recall"]))

#Print F1 Score
cat("\n Logistic Regression - F1 Score:", logistic_f1, "\n")
cat("XGBoost - F1 Score:", xgb_f1, "\n")

