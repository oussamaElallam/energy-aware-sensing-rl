import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Class labels
LABELS = ['Normal', 'Left BBB', 'Right BBB', 'Atrial Premature', 'PVC']

def load_data():
    """Load the test data"""
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    return X_test, y_test

def load_model():
    """Load the trained model"""
    model = tf.keras.models.load_model("ecg_classifier/ecg_model.h5")
    return model

def evaluate_model():
    """Perform comprehensive model evaluation"""
    # Load data
    print("Loading test data...")
    X_test, y_test = load_data()
    
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test[..., np.newaxis])
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    print("\nGenerating evaluation metrics...")
    
    # 1. Classification Report
    report = classification_report(y_test, y_pred, target_names=LABELS)
    print("\nClassification Report:")
    print(report)
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 3. ROC Curves
    plt.figure(figsize=(10, 8))
    for i in range(len(LABELS)):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{LABELS[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.close()
    
    # 4. Save detailed metrics to CSV
    metrics_dict = {
        'Class': LABELS,
        'Support': [sum(y_test == i) for i in range(len(LABELS))],
        'True Positives': [sum((y_pred == i) & (y_test == i)) for i in range(len(LABELS))],
        'False Positives': [sum((y_pred == i) & (y_test != i)) for i in range(len(LABELS))],
        'False Negatives': [sum((y_pred != i) & (y_test == i)) for i in range(len(LABELS))],
    }
    
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df['Precision'] = metrics_df['True Positives'] / (metrics_df['True Positives'] + metrics_df['False Positives'])
    metrics_df['Recall'] = metrics_df['True Positives'] / (metrics_df['True Positives'] + metrics_df['False Negatives'])
    metrics_df['F1 Score'] = 2 * (metrics_df['Precision'] * metrics_df['Recall']) / (metrics_df['Precision'] + metrics_df['Recall'])
    
    metrics_df.to_csv('detailed_metrics.csv', index=False)
    print("\nDetailed metrics have been saved to 'detailed_metrics.csv'")
    print("Visualizations have been saved as 'confusion_matrix.png' and 'roc_curves.png'")

if __name__ == "__main__":
    evaluate_model()
