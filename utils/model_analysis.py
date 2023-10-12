import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history, pathTosaveFig, fileName):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    sns.lineplot(data=history.history['accuracy'], label='Training Accuracy', ax=ax[0])
    sns.lineplot(data=history.history['val_accuracy'], label='Validation Accuracy', ax=ax[0])
    ax[0].set_title('Training and Validation Dataset Accuracy')
    ax[0].set_xlabel("Epochs --->")
    ax[0].set_ylabel("Accuracy --->")
    ax[0].legend()

    sns.lineplot(data=history.history['loss'], label='Training Loss', ax=ax[1])
    sns.lineplot(data=history.history['val_loss'], label='Validation Loss', ax=ax[1])
    ax[1].set_title('Training and Validation Dataset Loss')
    ax[1].set_xlabel("Epochs --->")
    ax[1].set_ylabel("Loss --->")
    ax[1].legend()
    
    plt.savefig(pathTosaveFig+fileName+'.png')
    plt.show()

def getClassificationReport(y_test_set, predict, target_names):
    return classification_report(y_test_set, predict, target_names = ['0','1'])

def predictionAnalysis(dataset, column_sentiment, column_1, pathTosaveFig, fileName):
    plt.figure(figsize=(7, 5))
    sns.set(font_scale=1.2)  # Increase font size of labels
    cm = confusion_matrix(dataset[column_sentiment], dataset[column_1])
    # Create heatmap
    sns.heatmap(
        cm, 
        cmap="Blues", 
        linecolor='white', 
        linewidths=1, 
        annot=True, 
        fmt='', 
        xticklabels=['negative', 'positive'], 
        yticklabels=['negative', 'positive']
    )

    # Set labels and title
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Adjust plot layout
    plt.tight_layout()
    plt.savefig(pathTosaveFig+fileName+'.png')
    plt.show()