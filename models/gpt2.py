import tensorflow as tf
import transformers
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from tokenizers import BertWordPieceTokenizer
from keras.layers import Dense,Input
import numpy as np
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import logging

def chooseTrained_Model_Tokenizer(config):
    pre_trained_model_name = config['pre_trained_model_name']
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pre_trained_model_name)
    # Load the GPT-2 model
    model = TFGPT2LMHeadModel.from_pretrained(pre_trained_model_name)
    fast_tokenizer = BertWordPieceTokenizer(config['fast_token_path'], lowercase=True)
    return model, tokenizer, fast_tokenizer

def encoder(texts, tokenizer, chunk_size=400):
    
    tokenizer.enable_truncation(max_length=200)
    tokenizer.enable_padding()
    tokens = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        tokens.extend([enc.ids for enc in encs])
    
    return np.array(tokens)


def build_model_for_training(transformer, config):
    
    input_word_tokens = Input(shape=(200,), dtype=tf.int32, name="input_word_tokens")
    sequence_output = transformer(input_word_tokens)[0]
    cls_token = sequence_output[:, 0, :]
    output = Dense(1, activation='sigmoid')(cls_token)
    
    # Unfreeze distilBERT layers and make available for training
    model = Model(inputs=input_word_tokens, outputs=output)
    for layer in model.layers:
      layer.trainable = False

    
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    # for layer in model.layers:
    #     layer.trainable = False
    #Define the checkpoints
    checkpoint_path = config['checkpoint_path']+"gpt_model_checkpoint.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch'
    )
    
    return model, checkpoint


def predict(dataset, fast_tokenizer, model):
    ecoded_data = encoder(dataset.values, fast_tokenizer)
    predict = model.predict(ecoded_data)
    predict_value = np.round(predict).astype(int)
    return predict_value



def getLogger():
    # create logger object
    logger = logging.getLogger('training_logs')
    logger.setLevel(logging.INFO)

    # create file handler and set formatter
    file_handler = logging.FileHandler('logs/gpt/training_logs.json')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger object
    logger.addHandler(file_handler)
    return logger

def maintainlog(logger, hist):
    logger.info({
    'model_name': 'gpt2',
    'accuracy': hist.history['accuracy'],
    'loss': hist.history['loss'],
    'val_accuracy': hist.history['val_accuracy'],
    'val_loss': hist.history['val_loss']
})