###CONFIG
#vectice_environment <- "base"
#reticulate::use_condaenv(condaenv = vectice_environment, required = TRUE)
#message(sprintf("using %s environment", vectice_environment))
#reticulate::py_config()
reticulate::py_install("vectice", pip = TRUE)
reticulate::py_module_available("vectice")
reticulate::py_module_available("pandas")

library(tidyverse)
library(caret)
library(dplyr)
library(randomForest)
library(gridExtra)

df_stroke_raw<-read.csv("RStudio Sample/healthcare-dataset-stroke-data.csv", header = TRUE)

# Drop the column with 'other'.(Since there is only 1 row)
df_stroke_clean = df_stroke_raw[!df_stroke_raw$gender == 'Other',]

#imputing dataset
df_stroke_clean$bmi[is.na(df_stroke_clean$bmi)]<- mean(df_stroke_clean$bmi,na.rm = TRUE)

df_stroke_clean$stroke<- factor(df_stroke_clean$stroke, levels = c(0,1), labels = c("No", "Yes"))
df_stroke_clean$gender<-as.factor(df_stroke_clean$gender)
df_stroke_clean$hypertension<- factor(df_stroke_clean$hypertension, levels = c(0,1), labels = c("No", "Yes"))
df_stroke_clean$heart_disease<- factor(df_stroke_clean$heart_disease, levels = c(0,1), labels = c("No", "Yes"))
df_stroke_clean$ever_married<-as.factor(df_stroke_clean$ever_married)
df_stroke_clean$work_type<-as.factor(df_stroke_clean$work_type)
df_stroke_clean$Residence_type<-as.factor(df_stroke_clean$Residence_type)
df_stroke_clean$smoking_status<-as.factor(df_stroke_clean$smoking_status)
df_stroke_clean$bmi<-as.numeric(df_stroke_clean$bmi)

write.csv(df_stroke_clean, "RStudio Sample/healthcare-dataset-stroke-data-clean.csv", row.names=FALSE)

#Lets split the final dataset to training and test data
n_obs <- nrow(df_stroke_clean)
split <- round(n_obs * 0.7)
train <- df_stroke_clean[1:split,]
# Create test
test <- df_stroke_clean[(split + 1):nrow(df_stroke_clean),]
write.csv(train, "RStudio Sample/train.csv", row.names=FALSE)
write.csv(test, "RStudio Sample/test.csv", row.names=FALSE)

#Modeling
set.seed(123)
rf_model<-randomForest(formula= stroke~.,data = train, metric='Accuracy', na.action=na.roughfix)
cm <- confusionMatrix(predict(rf_model, test), test$stroke)

######
#
# VECTICE DOCUMENTATION PIPELINE
#
######

vectice <- reticulate::import("vectice")
# Connect to Vectice
vct <- vectice$connect(config = 'vect_config.json')

#Catalog the raw data
phase <- vct$phase("PHA-1301")$create_iteration()
ds_resource <- vectice$FileResource(paths="RStudio Sample/healthcare-dataset-stroke-data.csv", dataframes = df_stroke_raw)
raw_ds <- vectice$Dataset$origin(name="Stroke_Dataset", resource=ds_resource)
phase$step_identify_data <- raw_ds
phase$step_describe <- "healthcare-dataset-stroke-data.csv is the source data for this project"
phase$complete()

#Catalog the cleaned data
phase <- vct$phase("PHA-1302")$create_iteration()
ds_resource <- vectice$FileResource(paths="RStudio Sample/healthcare-dataset-stroke-data-clean.csv", dataframes = df_stroke_clean)
clean_ds <- vectice$Dataset$clean(name="Stroke_Dataset_Clean", resource=ds_resource, attachments = "RStudio Sample/pty_id_states.jpg", derived_from = raw_ds$latest_version_id)
phase$step_clean_data <- clean_ds
phase$complete()

#Catalog the modeling dataset
phase <- vct$phase("PHA-1300")$create_iteration()
ds_train <- vectice$FileResource(paths="RStudio Sample/train.csv", dataframes = train)
ds_test <- vectice$FileResource(paths="RStudio Sample/test.csv", dataframes = test)
modeling_ds <- vectice$Dataset$modeling(name="Modeling_Dataset", training_resource=ds_train, testing_resource=ds_test, derived_from = clean_ds$latest_version_id)
phase$step_modeling_dataset <- modeling_ds

#Catalog the model itself
stats <- list("Accuracy" = cm$overall[1], "McnemarPValue" = cm$overall[7])
model <- vectice$Model(name = "Predictor model", library = "scikit-learn", technique = "randomForest", metrics = stats, attachments = "RStudio Sample/regression_graph.png", derived_from  = modeling_ds$latest_version_id)
phase$step_build_model <- model
phase$complete()
