#vectice_environment <- "base"
#reticulate::use_condaenv(condaenv = vectice_environment, required = TRUE)
#message(sprintf("using %s environment", vectice_environment))
#reticulate::py_config()
#reticulate::py_install("vectice", pip = TRUE)
#reticulate::py_module_available("vectice")
#reticulate::py_module_available("pandas")

library(tidyverse)

#Read the dataset
pty_id_raw <- read.csv("RStudio Sample/PTY_ID_MAIN.csv", header=TRUE)

# Basic Data Prep

#Remove extra header rows and CA customers from analysis
pty_id_rmhead <- subset(pty_id_raw,Customer_State_Address != 'Customer_State_Address')
pty_id_noCA <- subset(pty_id_rmhead,Customer_State_Address != 'CA')

#Drop Customer_Zip_Address variable to avoid bias trap and Customer_SSN and Customer_Email as they are PII variables
pty_id_clean <- subset(pty_id_noCA, select = -c(Customer_Zip_Address, Customer_SSN, Customer_Email))

# Save cleaned dataset locally
write.csv(pty_id_clean, "RStudio Sample/PTY_ID_CLEAN.csv", row.names=FALSE)

# Basic Data Visualization

#Plot customer count per states
pty_id_states_counts <- pty_id_clean %>%
  count(Customer_State_Address)
jpeg("RStudio Sample/pty_id_states.jpg", width = 800, height = "600")
pty_id_states_counts %>%
  mutate(Customer_State_Address = fct_reorder(Customer_State_Address, n, desc)) %>%
  ggplot(aes(x=Customer_State_Address, y=n)) +
  geom_col()
dev.off()

#######
#
# REST OF R CODDE FOR DATA CLEANING AND OTHER DATA PREPARATION PIPELINE
#
#######


#######
#
# EXAMPLE BUILD MODEL
#
#######

library(caret)
library(dplyr)
library(randomForest)
library(gridExtra)

df_stroke<-read.csv("RStudio Sample/healthcare-dataset-stroke-data.csv", header = TRUE)
# Drop the column with 'other'.(Since there is only 1 row)
df_stroke = df_stroke[!df_stroke$gender == 'Other',]
#imputing dataset
df_stroke$bmi[is.na(df_stroke$bmi)]<- mean(df_stroke$bmi,na.rm = TRUE)

df_stroke$stroke<- factor(df_stroke$stroke, levels = c(0,1), labels = c("No", "Yes"))
df_stroke$gender<-as.factor(df_stroke$gender)
df_stroke$hypertension<- factor(df_stroke$hypertension, levels = c(0,1), labels = c("No", "Yes"))
df_stroke$heart_disease<- factor(df_stroke$heart_disease, levels = c(0,1), labels = c("No", "Yes"))
df_stroke$ever_married<-as.factor(df_stroke$ever_married)
df_stroke$work_type<-as.factor(df_stroke$work_type)
df_stroke$Residence_type<-as.factor(df_stroke$Residence_type)
df_stroke$smoking_status<-as.factor(df_stroke$smoking_status)
df_stroke$bmi<-as.numeric(df_stroke$bmi)

#Lets split the final dataset to training and test data
n_obs <- nrow(df_stroke)
split <- round(n_obs * 0.7)
train <- df_stroke[1:split,]
# Create test
test <- df_stroke[(split + 1):nrow(df_stroke),]
write.csv(train, "RStudio Sample/train.csv", row.names=FALSE)
write.csv(test, "RStudio Sample/test.csv", row.names=FALSE)

#Modeling
set.seed(123)
rf_model<-randomForest(formula= stroke~.,data = train, metric='Accuracy', na.action=na.roughfix)

cm <- confusionMatrix(predict(rf_model, test), test$stroke)
str(cm)



######
#
# VECTICE DOCUMENTATION PIPELINE
#
######

vectice <- reticulate::import("vectice")
# Connect to Vectice
vct <- vectice$connect(config = 'vect_config.json')

#Catalog the raw data
#Start a new iteration
phase <- vct$phase("PHA-1301")$create_iteration()
ds_resource <- vectice$FileResource(paths="RStudio Sample/PTY_ID_MAIN.csv", dataframes = pty_id_rmhead)
raw_ds <- vectice$Dataset$origin(name="PTY_ID_MAIN", resource=ds_resource)
phase$step_identify_data <- raw_ds
phase$step_describe <- "PTY_ID_MAIN is the source data for this project"
phase$complete()

#Catalog the cleaned data
#Start a new iteration
phase <- vct$phase("PHA-1302")$create_iteration()
ds_resource <- vectice$FileResource(paths="RStudio Sample/PTY_ID_CLEAN.csv", dataframes = pty_id_clean)
clean_ds <- vectice$Dataset$clean(name="PTY_ID_CLEAN", resource=ds_resource, attachments = "RStudio Sample/pty_id_states.jpg", derived_from = raw_ds$latest_version_id)
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



