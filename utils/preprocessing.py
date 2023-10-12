import re
import nltk
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def checkForNull(dataset):
    """
    Checks for null values in the dataset and returns a string with the total number of null values for each column.

    Args:
        dataset (pandas.DataFrame): The dataset to check for null values.

    Returns:
        str: A string with the total number of null values in each column of the dataset.
    """
    # Compute the total number of null values for each column
    nullCheck = dataset.isnull().sum()

    # Return a string with the total number of null values for each column
    # Convert the integer value to a string before concatenating it with the string
    return 'Total null values in dataset: ' + nullCheck.astype(str)


def checkForDuplicated(dataset):
    """
    Checks if there are duplicate rows in the given dataset and prints the number of duplicate rows found.
    input params:
      dataset: a pandas DataFrame
    """
    duplicates = dataset.duplicated()
    print('Number of duplicate rows:',duplicates.sum())


def removeDuplicated(dataset):
    """
    This function takes a pandas dataframe as input that contains duplicate rows and 
    returns the same dataframe after removing those duplicate rows.
    
    Input params: 
      dataset: A pandas dataframe containing duplicate rows.

    Output params: 
      dataset : The input dataframe with duplicate rows removed.
    """
    dataset = dataset.drop_duplicates()
    print('Duplicates removed successfully')
    return dataset



def removeURL(dataframe, column):
    """
    Remove URLs from a pandas DataFrame column containing text reviews.
    input params:
        dataframe (pandas.DataFrame): The DataFrame to modify.
        column (str): The name of the column containing text reviews.
    output params:
        pandas.DataFrame: The modified DataFrame.
    """
    # Use a regular expression to remove URLs from the column
    dataframe[column] = dataframe[column].apply(lambda x: re.sub(r'http\S+', '', x))
    # Return the modified DataFrame
    return dataframe

def removeHtmlTags(dataframe, column):
    """
    Removes HTML tags from a pandas DataFrame column containing text reviews.
    intput params:
      dataframe : Pandas DataFrame The DataFrame containing the text reviews.
      column : The name of the column containing the text reviews.
    output params:
    pandas DataFrame: The modified DataFrame with HTML tags removed from the specified column.
    """
    # Remove HTML tags using regular expressions
    tags = re.compile('<.*?>')
    dataframe[column] = dataframe[column].apply(lambda x: re.sub(tags, '', x))
    return dataframe
  
def removeMentionsTags(dataframe, column):
    """
    Removes mentions tags from a pandas DataFrame column containing text reviews.
    input params:
      dataframe : Pandas DataFrame The DataFrame containing the text reviews.
      column : The name of the column containing the text reviews.
    output params:
      pandas DataFrame: The modified DataFrame with mentions tags removed from the specified column.
    """
    # Remove HTML tags using regular expressions
    mentions = re.compile(r'@\w+')
    dataframe[column] = dataframe[column].apply(lambda x: re.sub(mentions, '', x))
    return dataframe

def convert_to_lowercase(dataframe, column):
    """
    Convert text in a pandas DataFrame column to lowercase.
    intput params:
      dataframe : Pandas DataFrame The DataFrame containing the text reviews.
      column : The name of the column containing the text reviews.
    output params:
      pandas DataFrame: The modified DataFrame with lowercase text in the specified column.
    """
    dataframe[column] = dataframe[column].str.lower()
    return dataframe

def doCleaning(dataframe, column):
    """
    Clean text data in a pandas DataFrame column by removing URLs, HTML tags, mentions, and converting to lowercase.
    intput params:
      dataframe : Pandas DataFrame The DataFrame containing the text reviews.
      column : The name of the column containing the text reviews.
    output params:
      pandas DataFrame: The modified DataFrame with cleaned text in the specified column.
    """
    df = removeURL(dataframe, column)
    df = removeHtmlTags(df, column)
    df = removeMentionsTags(df, column)
    df = convert_to_lowercase(df, column)
    return df

def remove_stopwords(dataframe, column):
    """
    Remove stopwords from text in a pandas DataFrame column.
    intput params:
      dataframe : Pandas DataFrame The DataFrame containing the text reviews.
      column : The name of the column containing the text reviews.
    output params:
      pandas DataFrame: The modified DataFrame with stopwords removed from the specified column.
    """
    stop = stopwords.words('english')
    dataframe[column] = dataframe[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return dataframe



def removePunctuation(dataframe, column):
    """
    Remove punctuation from a pandas DataFrame column containing text reviews using regular expressions.
    intput params:
      dataframe: Pandas DataFrame containing the text reviews.
      column: The name of the column containing the text reviews.
    output params:
      Pandas DataFrame: The modified DataFrame with punctuation removed from the specified column.
    """
    # Remove punctuation using NLTK library
    puntuations = "[\.\?!,;:]"
    dataframe[column] = dataframe[column].apply(lambda x: re.sub(puntuations, '', x))
    return dataframe

def textLemmatization(dataframe, column):
    """
    Lemmatize words in a pandas DataFrame column containing text reviews using the WordNetLemmatizer from NLTK.
    input params:
      dataframe: Pandas DataFrame containing the text reviews.
      column: The name of the column containing the text reviews.
    output paramsturns:
      Pandas DataFrame: The modified DataFrame with lemmatized text in the specified column.
    """
    # Create lemmatizer object
    lemmatizer = WordNetLemmatizer()
    # Apply lemmatization to each word in review column
    dataframe[column] = dataframe[column].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    return dataframe

def performCleaning(dataframe, column):
    """
    Perform a series of text cleaning steps on a pandas DataFrame column containing text reviews.
    input params:
      dataframe: Pandas DataFrame containing the text reviews.
      column: The name of the column containing the text reviews.
    
    output params:
      Pandas DataFrame: The modified DataFrame with all specified text cleaning steps applied to the specified column.
    """
    # Perform each text cleaning step on the specified column
    df = removePunctuation(dataframe, column)
    df = textLemmatization(dataframe, column)
    df = remove_stopwords(df, column)
    return df