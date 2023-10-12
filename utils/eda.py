#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns

#############################
## Function to get dataset ##
#############################
def getDataSet(path):
    '''
    Load a CSV file into a Pandas DataFrame.
    Parameters:
        path (str): The path to the CSV file.
    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file.
    '''

    # Load the dataset from a CSV file
    df = pd.read_csv(path)

    # Return the dataset
    return df


######################################
## Function to describe the dataset ##
######################################
def decribeDataSet(path):
    '''
    This function reads a CSV file from a given path and returns a summary of the dataset.
    Parameters:
        path (str): The path of the CSV file
    Returns:
        pandas.DataFrame: A summary of the dataset
    
    '''
    # Load the dataset
    df = getDataSet(path)
    # Get summary of the dataset
    summary = df.describe()
    return summary


#####################################
## function to get the data counts ##
#####################################
def getDataCounts(path, column):
    '''
    Function to get count of each unique value in a given column of the dataset
    input params:
        path: path of csv file
        column: column name of the dataset for which we want to get value counts
    output params:
        sentiment_counts: pandas series object containing count of each unique value in the specified column

    '''
    # Load the dataset
    df = getDataSet(path)
    # Get the count of each unique value in the specified column
    sentiment_counts = df[column].value_counts()
    # Return the count of each unique value
    return sentiment_counts


#######################################
## function to plot bar chart in EDA ##
#######################################
def plotBarChart(data, xAxisLabel, YAxisLabels, Title, pathTosaveFig):
    '''
    Creates a bar plot from given data and saves it to a specified path.
    input params:
        data: Pandas Series or DataFrame with data to be plotted.
        xAxisLabel (str): Label for the x-axis.
        YAxisLabels (str): Label for the y-axis.
        Title (str): Title for the plot.
        pathTosaveFig (str): Path where the plot will be saved.
    output params:
        None
    '''
    # Create a bar plot using matplotlib
    plt.bar(data.index, data.values)

    # Add x-axis, y-axis, and title labels to the plot
    plt.xlabel(xAxisLabel)
    plt.ylabel(YAxisLabels)
    plt.title(Title)

    # Save the plot to the specified path
    plt.savefig(pathTosaveFig+'data_balanced_status.png')
    
    # Display the plot
    plt.show()


#######################################################################
## function to get samples for the early prediction without training ##
#######################################################################
def getSamples(path, NumberOfSamples):
    """
    This function is used to get the sample data from the dataset without training.
    input params:
        path: path of csv file
        NumberOfSamples: number of samples needed for early prediction
    output params:
        sample_data: panda dataframe containing the samples from the dataset
    """
    # Load the dataset from a CSV file
    df = getDataSet(path)
    
    # Get the specified number of samples from the dataset
    sample_data = df[0:NumberOfSamples]
    
    # Return the sample data
    return sample_data

####################################
## Function to convert the labels ##  
#################################### 
def convertLabelsTOBinary(dataset, column):
    ''' 
    Convert labels in a dataset column to binary values of 0 and 1
    input params:
        dataset : pandas dataframe Input dataset to convert
        column : Name of the column containing the labels to convert
    output params:
    pandas dataframe : The converted dataset with binary labels
    '''
    label_map = {'negative': 0, 'positive': 1}
    dataset[column] = dataset[column].map(label_map)
    return dataset


###################
### GPT2 - Model###
###################
def getGpt2Model():
    '''
    Loads pre-trained GPT-2 model and tokenizer for sequence classification
    output params:
        tokenizer: GPT-2 tokenizer for encoding input text
        model: pre-trained GPT-2 model for sequence classification
    
    '''
    # Load the pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2ForSequenceClassification.from_pretrained('gpt2')
    return tokenizer, model

#############################################################
## Define function to encode text and get GPT-2 embeddings ##
#############################################################
def get_gpt_embeddings(data, config):
    """
    Function to get the GPT-2 embeddings for the given text data.
    input params:
        data (list): The list of text data to get embeddings for.
        config (dict): A dictionary containing configuration parameters.
    output params:
        embeddings (torch.Tensor): A tensor containing the GPT-2 embeddings for the input text.

    """
    # Load the pre-trained GPT-2 model and tokenizer
    tokenizer, model = getGpt2Model()

     # Add special tokens for GPT-2
    tokenizer.add_special_tokens({'pad_token': config['tokenizer_pad_config']['pad']})
    
    # Tokenize text and convert to tensors
    inputs = tokenizer(
        data, 
        add_special_tokens=True, 
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    # Pass inputs through GPT-2 model
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get GPT-2 embeddings (last hidden state)
    embeddings = outputs[0]
    
    return embeddings


############################################################
## Define function to get sentiment from GPT-2 embeddings ##
############################################################
def predict_sentiment(dataset, config):
    """
    Predicts the sentiment of the given dataset using a pre-trained GPT-2 model and a linear layer.
    input params:
        dataset (str): A string of text representing the dataset.
        config (dict): A dictionary containing the configuration parameters for the GPT-2 model and the linear layer.
    output params:
        str: A string representing the predicted sentiment of the dataset. Either 'positive' or 'negative'.
    """
    # Get GPT-2 embeddings
    embeddings = get_gpt_embeddings(dataset, config)

    # Define linear layer to classify positive/negative sentiment
    linear = torch.nn.Linear(embeddings.shape[-1], 1)  # modify input size to match embedding tensor
    
    
    # Make predictions
    with torch.no_grad():
        logits = linear(embeddings)
        prediction = logits.sigmoid().round().squeeze().tolist()
    
    # Map predictions to 1.0 for positive and 0.0 for negative
    if prediction == 0.0:
      return 'negative'
    elif prediction == 1.0:
      return 'positive'
    else:
      return 'wrong prediction'



###################
### BERT - Model###
###################
def getBERTModel():
    """
    Load pre-trained BERT model and tokenizer.
    output params:
        tokenizer (BertTokenizer): a tokenizer from the pre-trained BERT model
        model (BertForSequenceClassification): a pre-trained BERT model for sequence classification
    """
    # Load pre-trained BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer, model


############################################################
## Define function to encode text and get BERT embeddings ##
############################################################
def get_bert_embeddings(text, config):
    """
    Get BERT embeddings (logits) for the input text.
    input params:
        text: A string of text to generate BERT embeddings for.
        config: A dictionary containing configuration parameters.
    output params:
        A PyTorch tensor of BERT embeddings for the input text.
    """
    # Load pre-trained BERT model and tokenizer
    tokenizer, model = getBERTModel()
    # Truncate input text to 512 tokens
    if len(text) > 512:
        text = text[:512]
    # Tokenize text and convert to tensors
    inputs = tokenizer(
        text, 
        add_special_tokens=True, 
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )
    
    # Pass inputs through BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get BERT embeddings (output logits)
    logits = outputs[0]
    
    return logits

###########################################################
## Define function to get sentiment from BERT embeddings ##
###########################################################
def predict_bert_sentiment(text, config):
    """
    Predict the sentiment of a given text using a pre-trained BERT model.
    input params:
        text (str): The input text to predict sentiment for.
        config (dict): A dictionary containing configuration parameters for the BERT model.
    output params:
        sentiment (str): The predicted sentiment of the input text as a string ('positive' or 'negative').
    """
    # Get BERT embeddings
    logits = get_bert_embeddings(text, config)
    
    # Example tensor with probabilities
    probs = torch.tensor(logits)
   
    # Get predicted class (0 for negative, 1 for positive)
    pred_class = torch.argmax(probs)
   
    # Get predicted label    
    predicted_label = torch.argmax(probs, dim=1).tolist()[0]
   
    # Map label to sentiment (0=negative, 1=positive)
    sentiment = 'positive' if predicted_label == 1 else 'negative'
    
    return sentiment

##################################
### Analysis B/w GPT2 and BERT ###
##################################
def evaluate_models(df, sentiment_col, pred_col):
    """
    This function evaluates the performance of a sentiment classification model by calculating accuracy, precision, recall
    and f1-score.
    input params:
        df: pandas DataFrame, a dataframe containing the sentiment and predicted sentiment columns.
        sentiment_col: str, the name of the column containing the true sentiment labels.
        pred_col: str, the name of the column containing the predicted sentiment labels.
    
    output params:
        metrics: dict, a dictionary containing the evaluation metrics (accuracy, precision, recall, f1-score).
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(df[sentiment_col], df[pred_col])
    precision = precision_score(df[sentiment_col], df[pred_col])
    recall = recall_score(df[sentiment_col], df[pred_col])
    f1 = f1_score(df[sentiment_col], df[pred_col])

    # Store metrics in dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics


#############################################
### Plot Metrics bar chart for comparison ###
#############################################

def plot_metrics(metrics_list, pathTosaveFig):
    """
    Plots a bar chart comparing the performance of two models based on different evaluation metrics.
    input params:
        metrics_list (list): A list of two dictionaries containing the evaluation metrics for two models.
        pathTosaveFig (str): The path where to save the generated plot.
    """
    # Define labels and scores for each metric for the two models
    labels = ['accuracy', 'precision', 'recall', 'f1']
    gpt_scores = [metrics_list[0][metric.lower()] for metric in labels]
    bert_scores = [metrics_list[1][metric.lower()] for metric in labels]
    x = np.arange(len(labels))
    width = 0.35
    
    # Plot the bar chart
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, gpt_scores, width, label='GPT')
    rects2 = ax.bar(x + width/2, bert_scores, width, label='BERT')
    
    # Add labels and legend
    ax.set_ylabel('Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add values on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.3f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    # save the plot
    plt.savefig(pathTosaveFig+'metrics.png')
    plt.show()


############################################
### Plot confusion metric for comparison ###
############################################
def plot_confusion_matrices(dataset, column_sentiment,column_1, column_2, pathTosaveFig):
    """
    Plots confusion matrices for two columns in a dataset.
    input params:
        dataset (pandas.DataFrame): DataFrame containing the data to plot.
        column_sentiment (str): Name of the column containing the true sentiment labels.
        column_1 (str): Name of the first column containing the predicted sentiment labels.
        column_2 (str): Name of the second column containing the predicted sentiment labels.
        pathTosaveFig (str): Path to save the resulting figure.
    """
    # Compute confusion matrix for GPT
    plot_1 = confusion_matrix(dataset[column_sentiment], dataset[column_1])
    
    # Compute confusion matrix for BERT
    plot_2 = confusion_matrix(dataset[column_sentiment], dataset[column_2])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot confusion matrix for GPT
    sns.heatmap(plot_1, annot=True, cmap='Blues', fmt='g', 
                xticklabels=['negative', 'positive'], yticklabels=['negative', 'positive'], 
                ax=axes[0])
    axes[0].set_xlabel('Predicted label')
    axes[0].set_ylabel('True label')
    axes[0].set_title('Confusion matrix for GPT')
    
    # Plot confusion matrix for BERT
    sns.heatmap(plot_2, annot=True, cmap='Blues', fmt='g', 
                xticklabels=['negative', 'positive'], yticklabels=['', ''], 
                ax=axes[1])
    axes[1].set_xlabel('Predicted label')
    axes[1].set_title('Confusion matrix for BERT')
    
    plt.tight_layout()
    plt.savefig(pathTosaveFig+'confusion_matrix.png')
    plt.show()


#####################################
### Plot ROC curve for comparison ###
#####################################
def plot_roc_curves(dataset, column_sentiment,column_1, column_2,pathTosaveFig):
    """
    Plots ROC curves for two models and saves the plot to a file.
    input params:
        dataset (pandas.DataFrame): the dataset containing the sentiment scores
        column_sentiment (str): the name of the column containing the true sentiment labels
        column_1 (str): the name of the column containing the predicted sentiment scores for model 1
        column_2 (str): the name of the column containing the predicted sentiment scores for model 2
        pathTosaveFig (str): the path to save the plot
    """
    # Compute ROC curve and AUC score for GPT model
    fpr_gpt, tpr_gpt, _ = roc_curve(dataset[column_sentiment], dataset[column_1])
    roc_auc_gpt = auc(fpr_gpt, tpr_gpt)

    # Compute ROC curve and AUC score for BERT model
    fpr_bert, tpr_bert, _ = roc_curve(dataset[column_sentiment], dataset[column_2])
    roc_auc_bert = auc(fpr_bert, tpr_bert)

    # Plot ROC curves side by side
    plt.figure(figsize=(8,6))
    plt.plot(fpr_gpt, tpr_gpt, color='darkorange', lw=2, label='GPT (AUC = %0.2f)' % roc_auc_gpt)
    plt.plot(fpr_bert, tpr_bert, color='blue', lw=2, label='BERT (AUC = %0.2f)' % roc_auc_bert)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(pathTosaveFig+'roc_curves.png')
    plt.show()