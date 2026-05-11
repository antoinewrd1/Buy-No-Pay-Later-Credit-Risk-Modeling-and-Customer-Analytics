jupyter nbconvert --to script app.ipynb

## ========================================================================
## 1. Install & Load Packages
## ========================================================================
packages <- c("tidyverse","caret","glmnet","randomForest","e1071",
              "rpart","dbscan","isotree","factoextra","nnet","pROC",
              "reshape2","kernlab")

installed <- rownames(installed.packages())
for (p in packages) {
  if (!(p %in% installed)) install.packages(p)
}

library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(e1071)
library(rpart)
library(dbscan)
library(isotree)
library(factoextra)
library(nnet)
library(pROC)
library(reshape2)
library(kernlab)

## ========================================================================
## 2. Load Data
## ========================================================================


df <- read.csv("data/Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset.csv")

## ========================================================================
## 3. Data Preparation
## ========================================================================
df <- df %>%
  mutate(
    employment_type = as.factor(employment_type),
    product_category = as.factor(product_category),
    location = as.factor(location),
    customer_segment = as.factor(customer_segment),
    default_flag = as.factor(default_flag),
    transaction_date = as.Date(transaction_date),
    days_since_transaction = as.numeric(Sys.Date() - transaction_date)
  ) %>%
  select(-transaction_date)

df$default_flag <- as.character(df$default_flag)
df$default_flag <- ifelse(df$default_flag %in% c("1","yes","Y","true","TRUE",1),
                          "default", "no_default")
df$default_flag <- factor(df$default_flag, levels = c("default","no_default"))

## ========================================================================
## 4. Exploratory Data Analysis (EDA)
## ========================================================================

# Risk Score Distribution
ggplot(df, aes(x = risk_score)) +
  geom_histogram(bins = 30, fill = "steelblue") +
  theme_minimal() +
  labs(title = "Distribution of Risk Score")

# Default Rate by Segment
ggplot(df, aes(x = customer_segment, fill = default_flag)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Default Rate by Segment", y = "Proportion")

# Income vs Risk
ggplot(df, aes(x = monthly_income, y = risk_score, color = default_flag)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Monthly Income vs Risk Score")

# Correlation Heatmap
num_df <- df %>% select(where(is.numeric))
corr_matrix <- cor(num_df, use = "complete.obs")
melted_corr <- melt(corr_matrix)

ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  theme_minimal() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white") +
  labs(title = "Correlation Heatmap")

## ========================================================================
## 5. Train/Test Split
## ========================================================================
set.seed(123)

train_index <- createDataPartition(df$default_flag, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]
train$default_flag <- factor(train$default_flag, levels = c("default","no_default"))
test$default_flag  <- factor(test$default_flag,  levels = c("default","no_default"))

## ========================================================================
## 6. Preprocessing
## ========================================================================
preProc <- preProcess(train %>% select(where(is.numeric)), method = c("center","scale"))
train_scaled <- predict(preProc, train %>% select(where(is.numeric)))
test_scaled  <- predict(preProc, test %>% select(where(is.numeric)))

## ========================================================================
## 7. Regression Models
## ========================================================================

# Linear Regression
lm_model <- lm(risk_score ~ ., data = train)
lm_pred <- predict(lm_model, test)
cat("Linear Regression RMSE:", RMSE(lm_pred, test$risk_score), "\n")

# Random Forest
rf_model <- randomForest(risk_score ~ ., data = train, ntree = 200)
rf_pred <- predict(rf_model, test)
cat("Random Forest RMSE:", RMSE(rf_pred, test$risk_score), "\n")

varImpPlot(rf_model)

## ========================================================================
## 8. Classification Models (Baseline)
## ========================================================================

# Decision Tree
tree_model <- rpart(default_flag ~ ., data = train, method = "class")
tree_pred <- predict(tree_model, test, type = "class")
cat("\nDecision Tree Confusion Matrix:\n")
print(confusionMatrix(tree_pred, test$default_flag))

# Logistic Regression
log_model <- glm(default_flag ~ ., data = train, family = "binomial")

prob_pred <- predict(log_model, test, type = "response")
class_pred <- factor(ifelse(prob_pred > 0.5, "default", "no_default"),
                     levels = c("default","no_default"))
cat("\nLogistic Regression (glm) Confusion Matrix:\n")
print(confusionMatrix(class_pred, test$default_flag))

## ========================================================================
## 9. Model Tuning (Cross-Validation)
## ========================================================================

control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Logistic (glm)
log_tuned <- train(default_flag ~ ., data = train,
                   method = "glm",
                   family = "binomial",
                   trControl = control,
                   metric = "ROC")

# Random Forest (classification)
rf_tuned <- train(default_flag ~ ., data = train,
                  method = "rf",
                  trControl = control,
                  tuneLength = 5,
                  metric = "ROC")

# SVM Radial
svm_tuned <- train(default_flag ~ ., data = train,
                   method = "svmRadial",
                   trControl = control,
                   tuneLength = 5,
                   metric = "ROC",
                   preProcess = c("center","scale"))

## ========================================================================
## 10. ROC Curves + AUC
## ========================================================================

log_probs <- predict(log_tuned, test, type = "prob")[, "default"]
rf_probs  <- predict(rf_tuned,  test, type = "prob")[, "default"]
svm_probs <- predict(svm_tuned, test, type = "prob")[, "default"]

y_true_num <- ifelse(test$default_flag == "default", 1, 0)

roc_log <- roc(y_true_num, log_probs)
roc_rf  <- roc(y_true_num, rf_probs)
roc_svm <- roc(y_true_num, svm_probs)

plot(roc_log, col = "blue", main = "ROC Curves")
plot(roc_rf, col = "red", add = TRUE)
plot(roc_svm, col = "green", add = TRUE)

legend("bottomright",
       legend = c(
         paste("Logistic AUC:", round(auc(roc_log),3)),
         paste("RF AUC:", round(auc(roc_rf),3)),
         paste("SVM AUC:", round(auc(roc_svm),3))
       ),
       col = c("blue","red","green"),
       lwd = 2)

# Model Comparison Table
results <- data.frame(
  Model = c("Logistic","Random Forest","SVM"),
  AUC = c(auc(roc_log), auc(roc_rf), auc(roc_svm))
)
print(results)

## ========================================================================
## 11. Clustering
## ========================================================================

set.seed(123)
kmeans_model <- kmeans(train_scaled, centers = 3, nstart = 25)

fviz_cluster(kmeans_model, data = train_scaled,
             ellipse.type = "norm",
             geom = "point",
             ggtheme = theme_minimal())

# Hierarchical
dist_mat <- dist(train_scaled)
hc_model <- hclust(dist_mat, method = "ward.D2")
plot(hc_model)

## DBSCAN
db <- dbscan(train_scaled, eps = 1.5, minPts = 5)
print(table(db$cluster))

## ========================================================================
## 12. Anomaly Detection
## ========================================================================

# Isolation Forest
iso_model <- isolation.forest(as.matrix(train_scaled))
scores <- predict(iso_model, as.matrix(train_scaled))

# One-Class SVM
svm_model <- svm(train_scaled, type = "one-classification", nu = 0.05)
svm_pred <- predict(svm_model, train_scaled)

## ========================================================================
## 13. Neural Network
## ========================================================================

train$default_flag <- factor(train$default_flag, levels = c("default","no_default"))
test$default_flag  <- factor(test$default_flag,  levels = c("default","no_default"))

nn_model <- nnet(default_flag ~ ., data = train,
                 size = 5, maxit = 200, trace = FALSE)

nn_prob <- as.numeric(predict(nn_model, test, type = "raw"))
nn_pred <- factor(ifelse(nn_prob > 0.5, "default", "no_default"),
                  levels = c("default","no_default"))

cat("\nNeural Network Confusion Matrix:\n")
print(confusionMatrix(nn_pred, test$default_flag))
