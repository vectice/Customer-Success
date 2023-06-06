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

#Drop Customer_Zip_Address variable to avoid bias trap and Customer_SSN as it is a PII variable
pty_id_clean <- subset(pty_id_noCA, select = -c(Customer_Zip_Address, Customer_SSN))

# Save cleaned dataset locally
write.csv(pty_id_clean, "RStudio Sample/PTY_ID_CLEAN.csv", row.names=FALSE)

#pty_id_clean %>%
#  View()

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


# VECTICE DOCUMENTATION PIPELINE
vectice <- reticulate::import("vectice")
pd <- reticulate::import("pandas")

# Connect to Vectice
vct <- vectice$connect(api_token = 'ynqdogKBN.6mDRJXaMkGjrobPx0LwvEynqdogKBNl69e8VYZO2dQ3WA47pgz')

#Catalog the raw data
#Start a new iteration
phase <- vct$phase("PHA-1280")$create_iteration()
pdDF <- reticulate::r_to_py(pty_id_raw)
ds_resource <- vectice$FileResource(paths="RStudio Sample/PTY_ID_MAIN.csv", dataframes = pdDF)
raw_ds <- vectice$Dataset$origin(name="PTY_ID_MAIN", resource=ds_resource)
phase$step_identify_data <- raw_ds
phase$step_describe <- "PTY_ID_MAIN is the source data for this project"
phase$complete()

#Catalog the cleaned data
#Start a new iteration
phase <- vct$phase("PHA-1281")$create_iteration()
pdDF <- reticulate::r_to_py(pty_id_clean)
ds_resource <- vectice$FileResource(paths="RStudio Sample/PTY_ID_CLEAN.csv", dataframes = pdDF)
clean_ds <- vectice$Dataset$clean(name="PTY_ID_CLEAN", resource=ds_resource, attachments = "RStudio Sample/pty_id_states.jpg")
phase$step_clean_data <- clean_ds
phase$complete()

