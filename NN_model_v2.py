# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 20:17:43 2020

@author: janpi

zadefinovanie jednotlivych modelov
"""

import tensorflow as tf

# funkcia vrati skompilovany model podla indexu
def create_RNN_model(neurons_in_layers,
                     model_index=1,
                     optimizer=tf.optimizers.RMSprop(),
                     loss_function='mae',
                     output_shape=1,
                     output_activation=None,
                     output_bias='zeros',
                     METRICS=[tf.metrics.MeanAbsoluteError(),
                           tf.metrics.MeanSquaredError()],
                     drop=0.2):
    """
    

    Parameters
    ----------
    neurons_in_layer : pocet neurovnov v jednotlivych vrstvach, musi to byt vektor
    ktory ma dlzku ako je pocet vrstiev
    model_index : TYPE, optional
        DESCRIPTION. The default is 1.
    optimizer : TYPE, optional
        DESCRIPTION. The default is 'RMSprop'.
    loss_function : TYPE, optional
        DESCRIPTION. The default is 'mae'.
    output_shape : kolko dni chceme predikovat, defaultne den do predu The default is 1.

    model_index = 0: jedna LSTM vrstva 
    model_index = 1: dve LSTM vrstvy
    model_index = 2: tri LSTM vrstvy
    model_index = 3: styri LSTM vrstvy
    model_index = 4: pat LSTM vrstiev
    model_index = 5: jedna LSTM vrstva, jedna Dense vrstva
    model_index = 6: dve LSTM vrstvy, jedna Dense vrstva
    Returns
    -------
    predvytvoreny model
    
    """
    
    if model_index == 0:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
    
    if model_index == 1:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[1],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
    
    if model_index == 2:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[1],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[2],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
        
    if model_index == 3:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[1],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[2],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[3],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
        
    if model_index == 4:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[1],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[2],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[3],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[4],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
        
    if model_index == 5:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=neurons_in_layers[1],
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123)),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
        
    if model_index == 6:
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(neurons_in_layers[0],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=True),
            
            tf.keras.layers.LSTM(neurons_in_layers[1],
                                 dropout=drop,
                                 kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                 return_sequences=False),
            
            tf.keras.layers.Dense(units=neurons_in_layers[2],
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123)),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
    
    if model_index == 7:
        model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(neurons_in_layers[0],
                                dropout=drop,
                                kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])
    if model_index == 8:
        model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(neurons_in_layers[0],
                                dropout=drop,
                                kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                return_sequences=True),
            
            tf.keras.layers.GRU(neurons_in_layers[1],
                                dropout=drop,
                                kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                return_sequences=False),
            
            tf.keras.layers.Dense(units=output_shape,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(seed=123),
                                  activation=output_activation, bias_initializer=output_bias)])    
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=METRICS)
    
    return(model)

# funkcia vrati pocet skrytych vrstiev
def return_length_of_hidden_layers(model_index):
    if model_index == 0:
        return(1)
    if model_index == 1:
        return(2)
    if model_index == 2:
        return(3)
    if model_index == 3:
        return(4)
    if model_index == 4:
        return(5)
    if model_index == 5:
        return(2)
    if model_index == 6:
        return(3)
    if model_index == 7:
        return(1)
    if model_index == 8:
        return(2)


