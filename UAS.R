getwd()
setwd("C:/komstat")
dataknn = read.csv("Customer_Churn.csv", sep = ",", header = TRUE)
dataknn = dataknn[,-1]
View(dataknn)

library(dplyr)
# Mengubah Gender menjadi numerik (0 = Female, 1 = Male)
dataknn = dataknn %>%
  mutate(Gender = ifelse(Gender == "Female", 0, 1))
# Mengubah Subscription Type menjadi numerik (1 = Basic, 2 = Standard, 3 = Premium)
dataknn = dataknn %>%
  mutate(Subscription.Type = as.numeric(factor(Subscription.Type, levels = c("Basic", "Standard", "Premium"))))
# Mengubah Contract Length menjadi numerik (1 = Monthly, 2 = Quarterly, 3 = Annual)
dataknn = dataknn %>%
  mutate(Contract.Length = as.numeric(factor(Contract.Length, levels = c("Monthly", "Quarterly", "Annual"))))
View(dataknn)

# EKSPLORASI DATA 
library(purrr)

# Fungsi untuk menghitung modus
mode_function = function(x) {
  unique_x = unique(x)
  unique_x[which.max(tabulate(match(x, unique_x)))]
}
# Menghitung modus 
mode_values = summarise_all(dataknn, list(mode = ~mode_function(.)))
mode_values

# Distribusi Data 
summary(dataknn)

# Visualisasi Data menggunakan Heatmap
library(corrplot)
par(mfrow=c(1,1))
num_data = dataknn[, sapply(dataknn, is.numeric)]
cor_matrix = cor(num_data, use = "complete.obs")
# Membuat visualisasi korelasi dengan corrplot
par(mfrow = c(1, 1))  
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", 
         tl.srt = 45, 
         addCoef.col = "black",
         number.cex = 0.7) 

# Data Cleaning
# Menghitung persentase missing values per kolom
na_percentage = sapply(dataknn, function(x) sum(is.na(x)) / length(x) * 100)
print(na_percentage)
# Menghapus baris dengan missing values
dataknn_cleaned = na.omit(dataknn)
# Mengubah tipe data 'Churn' menjadi faktor (categorical variable)
dataknn_cleaned$Churn = as.factor(dataknn_cleaned$Churn)
# Menampilkan ringkasan data setelah data cleaning
summary(dataknn_cleaned)
# Menampilkan jumlah baris setelah penghapusan missing values
cat("Jumlah baris setelah dilakukan data cleaning:", nrow(dataknn_cleaned), "\n")

# Split Dataset
set.seed(123)
library(caret)
# Membagi dataset menjadi 80% untuk training dan 20% untuk testing
index = createDataPartition(dataknn_cleaned$Churn, p = 0.8, list = FALSE)
train_data = dataknn_cleaned[index, ]
test_data = dataknn_cleaned[-index, ]
# Menampilkan dimensi data training dan testing
cat("Dimensi data training:", nrow(train_data), "\n")
cat("Dimensi data testing:", nrow(test_data), "\n")
# Menampilkan distribusi kelas 'Churn' di data training dan testing
cat("Distribusi kelas pada data training:\n")
print(table(train_data$Churn))
cat("Distribusi kelas pada data testing:\n")
print(table(test_data$Churn))

# Analisis Data dengan Metode K-Nearest Neighbors (KNN)
library(lattice)
library(ggplot2)
library(caret)

# Mengubah kolom Churn menjadi faktor 
train_data$Churn <- factor(train_data$Churn)
test_data$Churn <- factor(test_data$Churn)
# Mengatur cross-validation untuk 5 fold
crosval <- trainControl(method = "cv", number = 5)
# Melatih model KNN dengan tuning parameter k
data_knn <- train(
  Churn ~ ., data = train_data, method = 'knn',
  tuneGrid = expand.grid(k = 1:20),
  trControl = crosval, preProcess = c('center', 'scale')
)
# Melakukan prediksi pada data testing
prediction <- predict(data_knn, newdata = test_data)
# Menyesuaikan level faktor pada data testing
test_data$Churn <- factor(test_data$Churn, levels = levels(train_data$Churn))
prediction <- factor(prediction, levels = levels(train_data$Churn))
# Membuat confusion matrix untuk evaluasi model
confusion_matrix <- confusionMatrix(prediction, test_data$Churn)
# Menampilkan confusion matrix
print(confusion_matrix)
# Plot grafik akurasi berdasarkan nilai k
plot(data_knn, main = "Grafik Akurasi Nilai K-KNN", xlab = "Nilai K-KNN")

# Analisis Data dengan Metode Random Forest 
library(randomForest)
library(ggplot2)
# Memastikan label Churn menjadi faktor pada train_data dan test_data
train_data$Churn <- as.factor(train_data$Churn)
test_data$Churn <- as.factor(test_data$Churn)
# Membangun model Random Forest
model_rf <- randomForest(Churn ~ ., data = train_data, ntree = 100, mtry = 2, importance = TRUE)
# Melakukan prediksi 
predictions <- predict(model_rf, test_data)
# Membuat confusion matrix
confusion_matrix <- table(predictions, test_data$Churn)
print(confusion_matrix)
# Menghitung akurasi model
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
sprintf("Akurasi: %.2f%%", accuracy * 100)
print(importance(model_rf))
# Membuat confusion matrix sebagai dataframe
conf_matrix <- as.data.frame(as.table(confusion_matrix))
colnames(conf_matrix) <- c("Predicted", "Actual", "Freq")
# Membuat heatmap
ggplot(data = conf_matrix, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "yellow", size = 5) +
  scale_fill_gradient(low = "darkred", high = "black") +
  labs(title = "Heatmap Confusion Matrix", x = "Prediksi", y = "Label Sebenarnya") +
  theme_minimal()



