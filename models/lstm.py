import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import logging
import numpy as np

#######################
## Define LSTM model ##
#######################
def create_model(vocab_size, max_len, config):
    """
    Creates a Sequential model with an Embedding layer, LSTM layer, and Dense layer.
    input params:
        vocab_size (int): The size of the vocabulary.
        max_len (int): The maximum length of the input sequence.
        config (dict): A dictionary containing the configuration for the model.
    output params:
        model (Sequential): The compiled model.
        checkpoint (ModelCheckpoint): A callback that saves the best weights of the model during training.
    """
    # Create a Sequential model
    model = Sequential()
    
    # Add an Embedding layer to the model with the specified input_dim, output_dim, and input_length
    model.add(Embedding(input_dim=vocab_size, output_dim=config['embedding']['output_dim'], input_length=max_len))

    # Add an LSTM layer to the model with the specified units, dropout, recurrent_dropout, activation, and recurrent_activation
    model.add(LSTM(units=config['lstm']['units'], 
                   dropout=config['lstm']['dropout'], 
                   recurrent_dropout=config['lstm']['recurrent_dropout'],
                   activation=config['lstm']['activation'],
                   recurrent_activation = config['lstm']['recurrent_activation']
                   ))
    
    # Add a Dense layer to the model with the specified units and activation
    model.add(Dense(units=config['dense']['units'], activation=config['dense']['activation']))
    
    # Compile the model with the specified optimizer, loss, and metrics
    model.compile(optimizer=config['optimizer'], loss=config['loss'], metrics=[config['metrics']])
    
    # Set up a callback to save the best weights of the model during training
    checkpoint_path = config['checkpoint_path']+"lstm_model.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch'
    )
    # Return the compiled model and the checkpoint callback
    return model, checkpoint

# Train LSTM model
def train_model(model, X_train, y_train, X_val, y_val, epochs,  batchsize, callbacks):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batchsize, callbacks=callbacks)
    # Define the checkpoint
    
    return history

def convertTextTotoken(data, config):
    """
    Converts input text data into tokens using Keras Tokenizer.
    input params:
        data (list): A list of text data.
        config (dict): A dictionary containing the tokenization parameters.

    output params:
        tokens (ndarray): A 2D numpy array of shape (num_samples, maxlen) where num_samples is the number of samples
                            in the input data and maxlen is the maximum length of the sequence.
    """
    # Load and preprocess data
    tokenizer = Tokenizer(config['tokenization']['num_words'])
    tokenizer.fit_on_texts(data)

    tokens = tokenizer.texts_to_sequences(data)
    tokens = pad_sequences(tokens, maxlen=config['tokenization']['maxlen'])

    # return tokes
    return tokens

############################################
## log detail of training in the log file ##
############################################
def getLogger():
    """
    Creates and returns a logger object that writes log messages to a file.
    output params:
        logger: A logger object that writes log messages to a file.
    """
    # create logger object
    logger = logging.getLogger('training_logs')
    logger.setLevel(logging.INFO)

    # create file handler and set formatter
    file_handler = logging.FileHandler('logs/lstm/training_logs.json')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger object
    logger.addHandler(file_handler)
    return logger

#################################
## Update logs in the log file ##
#################################
def maintainlog(logger, hist):
    """
    A helper function to maintain the training logs.
    input params:
        logger (logging.Logger): The logger object to use for logging.
        hist (keras.callbacks.History): The Keras history object containing the training metrics.
    """
    # Log the training metrics to the logger object
    logger.info({
    'model_name': 'lstm',
    'accuracy': hist.history['accuracy'],
    'loss': hist.history['loss'],
    'val_accuracy': hist.history['val_accuracy'],
    'val_loss': hist.history['val_loss']
})

##########################################
## Predict the results on trained model ##
##########################################
def predict(model, test_data):
    # Make predictions on test data
    y_pred = model.predict(test_data)
    predict_value = np.round(y_pred).astype(int)
    # Convert predicted probabilities to predicted labels

    return predict_value
