import sys
import numpy as np
import json
import vectice
import pandas as pd


def main(argv):
    # TODO: parse the arguments passed using argparse instead of below
    # NEEDS error handling
    config_path = argv[1]
    config_data = json.load(open(config_path))
    
    data_path = config_data["data_path"]
    data_target = config_data["target"]
    phase = config_data["phase_id"]
    api_token = config_data["api_token"]
    
    dataset_id = None
    if "dataset_id" in config_data:
        dataset_id = config_data["dataset_id"]
        
    model_id = None
    if "model_id" in config_data:
        model_id = config_data["model_id"]
        
    
    vct_conn = vectice.connect(api_token=api_token)
    vct_iter = vct_conn.phase(phase).create_or_get_current_iteration()
    
    if model_id is not None:
        updateVersion(vct_conn, model_id)
    else:
        clean, test, train, test_target = data_split(data_path)
        trgt_dist = data_xploration(clean, data_target)
        clean_widget = log_clean_ds(clean, data_path, trgt_dist)
        vct_iter.log(clean_widget)
        
        train_no_missing, train_labels, test_no_missing  = data_encoding(train, test)
        
        modeling_widget = log_modeling_ds(clean,train_no_missing, test_no_missing, data_path, clean_widget.latest_version_id)
        vct_iter.log(modeling_widget)
        
        model, metric, figs = modeling(train_no_missing, train_labels, test_no_missing, test_target)
        model_widget = log_model("Probability of Default", model, metric,"Random Forest", figs, modeling_widget.latest_version_id )
        vct_iter.log(model_widget)
        vct_iter.complete()        
        
        

def log_modeling_ds(clean, train, test, data_path, derived):
    from vectice import Dataset, NoResource
    modeling_ds = Dataset.modeling(name = "Modeling",training_resource = NoResource(dataframes=train, origin=""), 
                            testing_resource=NoResource(dataframes=test, origin=""),
                            derived_from=derived)
    return modeling_ds
    
def log_clean_ds(clean, data_path, att):
    from vectice import Dataset, FileResource
    data_clean = Dataset.clean(name = "Clean Dataset", resource = FileResource(paths=data_path, dataframes=clean), attachments=att)
    return data_clean
    
def log_model(name, model, metric, technique, figs, derived):
    from vectice import Model
    model = Model(name=name, 
                      predictor=model, 
                      properties = model.get_params(), 
                      metrics = metric, 
                      attachments=figs,
                      derived_from=derived,
                      technique=technique)
    return model

def updateVersion(conn, widget_id):
    widget_version = conn.browse(widget_id)
    widget_version.update(attachments=["roc_curve.png"])
    
def data_split(data_path):
    from sklearn.model_selection import train_test_split
    # Training data
    application_cleaned = pd.read_csv(data_path)
    app_train_feat, app_test_feat = train_test_split(application_cleaned, test_size=0.15, random_state=42)
    # Separate the target variable from the testing set
    target_variable = 'TARGET'
    app_test_feat_target = app_test_feat[target_variable]
    app_test_feat = app_test_feat.drop(target_variable, axis=1)

    # Print the shapes of the resulting dataframes
    #print('Training data shape: ', app_train_feat.shape)
    #print('Testing shape: ', app_test_feat.shape)
    #print('Testing target shape: ', app_test_feat_target.shape)
    return application_cleaned, app_test_feat, app_train_feat, app_test_feat_target
    
def data_xploration(df, target):
    import matplotlib.pyplot as plt
    df[target].astype(int).plot.hist()
    plt.tight_layout()
    plt.savefig("Target distribution.png")
    return("Target distribution.png")

def data_encoding(train, test):
    # sklearn preprocessing for dealing with categorical variables
    from sklearn.preprocessing import LabelEncoder
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in train:
        if train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(train[col].unique())) <= 2:
                # Train on the training data
                le.fit(train[col])
                # Transform both training and testing data
                train[col] = le.transform(train[col])
                #train[col] = le.transform(train[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1
                
    #return '%d columns were label encoded.' % le_count
    # one-hot encoding of categorical variables
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    train_labels = train['TARGET']

    # Align the training and testing data, keep only columns present in both dataframes
    train, test = train.align(test, join = 'inner', axis = 1)

    # Add the target back in
    train['TARGET'] = train_labels


    # Create an anomalous flag column
    train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


    test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
    test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
    #print('Training Features shape: ', train.shape)
    #print('Testing Features shape: ', test.shape)
    #print('There are %d anomalies in the test data out of %d entries' % (test["DAYS_EMPLOYED_ANOM"].sum(), len(test)))
    
    # Age information into a separate dataframe
    #train['DAYS_BIRTH'] = abs(train['DAYS_BIRTH'])
    #train['YEARS_BIRTH'] = train['DAYS_BIRTH'] / 365

    # Bin the age data
    #train['YEARS_BINNED'] = pd.cut(train['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    
    from sklearn.impute import SimpleImputer

    # Drop the target from the training data
    if 'TARGET' in train:
        train_no_missing = train.drop(columns = ['TARGET'])

    features = list(train_no_missing.columns)
    #print(train_no_missing)
    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')
    # Fit on the training data
    imputer.fit(train_no_missing)

    # Transform both training and testing data
    train_no_missing = pd.DataFrame(imputer.transform(train_no_missing), columns=features).set_index('SK_ID_CURR')
    test_no_missing = pd.DataFrame(imputer.transform(test), columns=features).set_index('SK_ID_CURR')
    
    return train_no_missing, train_labels, test_no_missing

def modeling(train_no_missing, train_labels, test_no_missing, app_test_feat_target):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, roc_curve, auc
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    # Make the random forest classifier
    random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, min_samples_leaf=0.01, n_jobs = -1)
    features = list(train_no_missing.columns)
    # Train on the training data
    random_forest.fit(train_no_missing, train_labels)

    # Extract feature importances
    feature_importance_values = random_forest.feature_importances_

    # Make predictions on the test data
    predictions = random_forest.predict_proba(test_no_missing)[:, 1]

    roc_auc = roc_auc_score(app_test_feat_target.values, predictions)
    # Sort instances based on predicted probabilities
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_labels = app_test_feat_target.iloc[sorted_indices]

    # Define the desired percentage (e.g., 25%)
    desired_percentage = 0.25

    # Identify the threshold probability corresponding to the desired percentage
    threshold_index = int(desired_percentage * len(predictions))
    threshold_probability = predictions[sorted_indices[threshold_index]]

    # Apply the threshold to classify instances
    binary_predictions = (predictions >= threshold_probability).astype(int)

    # Calculate the recall at the desired percentage
    recall = recall_score(app_test_feat_target.values, binary_predictions)
    f1 = f1_score(app_test_feat_target.values,binary_predictions)

    metric = {"auc": float(roc_auc),
            f"recall at {desired_percentage}%": float(recall),
            f"f1_score at {desired_percentage}%":float(f1)}

    #print("ROC AUC Score:", roc_auc)
    #print("F1 Score:", f1)
    #print("Recall Score:", recall)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(app_test_feat_target.values, predictions)
    roc_auc = auc(fpr, tpr)
    
   

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")

    # Plot calibration curve
    prob_true, prob_pred = calibration_curve(app_test_feat_target.values, predictions, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Random Forest', linestyle='-', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc="upper left")
    plt.savefig("calibration.png")

    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=30, color='blue', alpha=0.7)
    plt.title('Distribution of Predicted Probabilities')
    plt.xlabel('Predicted Probability')
    plt.savefig("Prediction_distribution.png")
    plt.ylabel('Frequency')

    # Create a DataFrame with predicted probabilities and true labels
    df_results = pd.DataFrame({'Probability': predictions, 'Default': app_test_feat_target.values})

    # Sort instances based on predicted probabilities
    df_results = df_results.sort_values(by='Probability', ascending=False)

    # Divide the sorted instances into quantiles (e.g., deciles)
    num_quantiles = 10
    df_results['Quantile'] = pd.qcut(df_results['Probability'], q=num_quantiles, labels=False, duplicates='drop')

    # Calculate the percentage of defaults in each quantile
    quantile_defaults = df_results.groupby('Quantile')['Default'].mean() * 100

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(quantile_defaults.index, quantile_defaults.values, color='blue', alpha=0.7)
    plt.xlabel('Quantile of predicted probabilities')
    plt.ylabel('Percentage of Defaults')
    plt.title('Percentage of Defaults by Quantile of Predicted Probabilities')
    plt.xticks(ticks=quantile_defaults.index, labels=[f'Q{i + 1}' for i in quantile_defaults.index])
    plt.savefig("Percentage of Defaults by Quantile.png")
    
    figs=["roc_curve.png","calibration.png","Prediction_distribution.png", "Percentage of Defaults by Quantile.png"]
    
    return random_forest, metric, figs
    
    

    
##############################################################
# Script main entry point
if __name__ == '__main__':
    main(sys.argv)