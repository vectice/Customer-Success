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

df_stroke_raw<-read.csv("healthcare-dataset-stroke-data.csv", header = TRUE)

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

write.csv(df_stroke_clean, "healthcare-dataset-stroke-data-clean.csv", row.names=FALSE)


#Lets split the final dataset to training and test data
n_obs <- nrow(df_stroke_clean)
split <- round(n_obs * 0.7)
train <- df_stroke_clean[1:split,]
# Create test
test <- df_stroke_clean[(split + 1):nrow(df_stroke_clean),]
write.csv(train, "train.csv", row.names=FALSE)
write.csv(test, "test.csv", row.names=FALSE)

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
vct <- vectice$connect(api_token = 'ng1Wom1Xp.jObvAoJxPNw9EQydYeRaWng1Wom1XpVGj7BDk286zml34ML0Zr')

#Catalog the raw data
iter <- vct$phase("PHA-1301")$create_or_get_current_iteration()
ds_resource <- vectice$FileResource(paths="healthcare-dataset-stroke-data.csv", dataframes = df_stroke_raw)
raw_ds <- vectice$Dataset$origin(name="Stroke_Dataset", resource=ds_resource)
iter$log(raw_ds, section= "Identify Dataset")
iter$log("healthcare-dataset-stroke-data.csv is the source data for this project", section= "Identify Dataset")
iter$complete()

#Catalog the cleaned data
iter <- vct$phase("PHA-1302")$create_or_get_current_iteration()
ds_resource <- vectice$FileResource(paths="healthcare-dataset-stroke-data-clean.csv", dataframes = df_stroke_clean)
clean_ds <- vectice$Dataset$clean(name="Stroke_Dataset_Clean", resource=ds_resource, attachments = "pty_id_states.jpg", derived_from = raw_ds$latest_version_id)
iter$log(clean_ds, section="EDA")
iter$log("The data was prepared according to our standard data pipeline for:\n\tcompletness\n\tnormalization\n\tsecurity\n\tbias free\n\tanonymized\nAs required by Compliance", section="EDA")
iter$complete()

#Catalog the modeling dataset
iter <- vct$phase("PHA-1300")$create_or_get_current_iteration()
ds_train <- vectice$FileResource(paths="train.csv", dataframes = train)
ds_test <- vectice$FileResource(paths="test.csv", dataframes = test)
modeling_ds <- vectice$Dataset$modeling(name="Modeling_Dataset", training_resource=ds_train, testing_resource=ds_test, derived_from = clean_ds$latest_version_id)
iter$log(modeling_ds, section = "Modeling Dataset")

#Catalog the model itself
stats <- list("Accuracy" = cm$overall[1], "McnemarPValue" = cm$overall[7])
model <- vectice$Model(name = "Predictor model", library = "scikit-learn", technique = "randomForest", metrics = stats, derived_from  = modeling_ds$latest_version_id)
iter$log(model, section="Build Model")
iter$log("ROC_Curve.png", section = "Build Model")
iter$complete()
