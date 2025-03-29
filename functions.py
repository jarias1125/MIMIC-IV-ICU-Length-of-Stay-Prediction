# File with all the functions we need in our analysis

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, classification_report, brier_score_loss, precision_recall_curve, auc, roc_curve
from sklearn.calibration import calibration_curve
from netcal.metrics import ECE
from sklearn.model_selection import GridSearchCV

# Clean the dataset
def clean_df(df):
    df = df.dropna() 
    df = df.drop_duplicates()  
    df = df[df['lengths_of_1st_admission'] >= 0]   # removing patients with first LOS admission < 0 == patients who die before they arrive at the hospital
    return df


# Evaluate model performance
def model_evaluation(y_true, y_pred, y_prob):

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()  # true positive/false positive/false negative/true negative
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc_score = auc(recall, precision)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')  # 'quantile' to ensure that each bin has the same number of samples (no empty bins)

    # Compute Expected Calibration Error (ECE)
    bin_counts, _ = np.histogram(y_prob, bins=np.linspace(0, 1, 10 + 1))
    total_samples = len(y_true)
    if len(prob_pred) < len(bin_counts):      
        bin_counts = bin_counts[:len(prob_pred)]
    ece_score = np.sum((bin_counts / total_samples) * np.abs(prob_pred - prob_true))

    # Key performance metrics for binary classification
    metrics = {
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred),4),
        'Recall': round(recall_score(y_true, y_pred),4),
        'F1 Score': round(f1_score(y_true, y_pred),4),
        'Brier Score Loss': round(brier_score_loss(y_true, y_prob),4),
        'AUC (ROC)': round(roc_auc_score(y_true, y_pred),4),
        'AUC (PRC)': round(auprc_score, 4),
        'ECE' : round(ece_score, 4)
    }

    print(f'Classification report:\n {classification_report(y_true, y_pred)}')

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Print some evaluation metrics
    print(f'Brier Score Loss: {metrics["Brier Score Loss"]:.3f}')
    print(f'AUROC: {metrics["AUC (ROC)"]:.3f}')
    print(f'AUPRC: {metrics["AUC (PRC)"]:.3f}')

    # Plot calibration curve
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', linestyle='-', label='Model')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label="Perfectly Calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid()
    plt.show()

    return metrics


# Train and evaluate the model 
def model_train(model, param_grid, x_train, x_test, y_train, y_test):
    # Search for hyperparameters 
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)    
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_     # Save the best model parameters
    y_pred = best_model.predict(x_test)
    y_prob = best_model.predict_proba(x_test)[:, 1]
    
    # evaluate the model performance
    metrics = model_evaluation(y_test, y_pred, y_prob)
    results = { 
        "model": best_model,
        "best_params": grid_search.best_params_
    }
    return results, metrics





