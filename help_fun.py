# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 21:18:36 2020

@author: janpi

pomocne funkcie na normalizaciu dat a vytrvaranie datasetov
"""

import talib
import numpy as np
import pandas as pd


def normalize_data(data, split=None, normalization='min_max'):
    """
    normalizuje data, bud min_max normalizacia alebo standardizacia.
    parametre pre normalizaciu pocita bud z celeho datasetu, alebo len z [:split]

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    split : TYPE, optional
        DESCRIPTION. The default is None.
    normalization : TYPE, optional
        DESCRIPTION. The default is 'min_max'.

    Returns
    -------
    None.

    """
    if not (normalization == 'min_max' or normalization == 'standardization'):
        return('normalization must be specified as min_max or standardization')
    if split is None:
        if normalization == 'min_max':
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            data = (data - data_min) / (data_max - data_min)
            return(data, data_min, data_max)
        elif normalization == 'standardization':
            data_mean = data.mean(axis=0)
            data_std = data.std(axis=0)
            data = (data - data_mean) / data_std
            return(data, data_mean, data_std)
    else:
        if normalization == 'min_max':
            data_min = data[:split].min(axis=0)
            data_max = data[:split].max(axis=0)
            data = (data - data_min) / (data_max - data_min)
            return(data, data_min, data_max)
        elif normalization == 'standardization':
            data_mean = data[:split].mean(axis=0)
            data_std = data[:split].std(axis=0)
            data = (data - data_mean) / data_std
            return(data, data_mean, data_std)


def normalize_each_subset_alone(data, train_split, val_split, normalization='min_max'):
    """
    normalizuje kazdu sadu samostatne

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    train_split : TYPE
        DESCRIPTION.
    val_split : TYPE
        DESCRIPTION.
    normalization : TYPE, optional
        DESCRIPTION. The default is 'min_max'.

    Returns
    -------
    None.

    """
    if not (normalization == 'min_max' or normalization == 'standardization'):
        return('normalization must be specified as min_max or standardization')
    if normalization == 'min_max':
        data_train_min = data[:train_split].min(axis=0)
        data_train_max = data[:train_split].max(axis=0)
        data[:train_split] = (data[:train_split] - data_train_min) / (data_train_max - data_train_min)
        
        data_val_min = data[train_split:val_split].min(axis=0)
        data_val_max = data[train_split:val_split].max(axis=0)
        data[train_split:val_split] = (data[train_split:val_split] - data_val_min) / (data_val_max - data_val_min)
        
        data_test_min = data[val_split:].min(axis=0)
        data_test_max = data[val_split:].max(axis=0)
        data[val_split:] = (data[val_split:] - data_test_min) / (data_test_max - data_test_min)
        return(data, data_train_min, data_train_max, data_val_min, data_val_max, data_test_min, data_test_max)
    elif normalization == 'standardization':
        data_train_mean = data[:train_split].mean(axis=0)
        data_train_std = data[:train_split].std(axis=0)
        data[:train_split] = (data[:train_split] - data_train_mean) / data_train_std
        
        data_val_mean = data[train_split:val_split].mean(axis=0)
        data_val_std = data[train_split:val_split].std(axis=0)
        data[train_split:val_split] = (data[train_split:val_split] - data_val_mean) / data_val_std
        
        data_test_mean = data[val_split:].mean(axis=0)
        data_test_std = data[val_split:].std(axis=0)
        data[val_split:] = (data[val_split:] - data_test_mean) / data_test_std
        return(data, data_train_mean, data_train_std, data_val_mean, data_val_std, data_test_mean, data_test_std)


def denormalize_data(data, normalization='min_max', logaritmization=False, data_max=None, data_min=None, data_mean=None, data_std=None):
    """
    zo znormalizovanych dat vrati povodne

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    normalization : TYPE, optional
        DESCRIPTION. The default is 'min_max'.
    logaritmization : TYPE, optional
        DESCRIPTION. The default is False.
    data_max : TYPE, optional
        DESCRIPTION. The default is None.
    data_min : TYPE, optional
        DESCRIPTION. The default is None.
    data_mean : TYPE, optional
        DESCRIPTION. The default is None.
    data_std : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if not (normalization == 'min_max' or normalization == 'standardization'):
        return('normalization must be specified as min_max or standardization')
    if normalization == 'min_max':
        if logaritmization:
            true_data = data * (data_max - data_min) + data_min
            true_data = np.exp(true_data)
        else:
            true_data = data * (data_max - data_min) + data_min
    
    if normalization == 'standardization':
        if logaritmization:
            true_data = data * data_std + data_mean
            true_data = np.exp(true_data)
        else:
            true_data = data * data_std + data_mean
    
    return(true_data)


def univariate_data(datasets,
                    start_index, end_index,
                    history_size, target_size):
    '''
    Parameters
    ----------
    datasets : vstupne data, napriklad dane pozorovania ktore su vekrove
                (nie matica)
    start_index : index od ktoreho zacina vybrana mnozina dat z vektora napr.
                ak delime data na trenovacie, validacne a testovacie tak start
                index posuvame podla toho
    end_index : podobne ako start_index, len oznacuje koniec
    history_size : velkost vzorky, ktora bude tvorit data, pomocou ktorych
                    predikujeme (velkost informacii, ktore mame)
    target_size : o kolko dopredu chceme predikovat
    Returns
    -------
    vracia dve np.array data ->>> ktory obsahuje end_index - start_index - history_size
    vzoriek lebo tolko vieme vyrobit z danej mnoziny a kazda ma rozmer (hist_size, 1)
    a label ktore maju velkost (pocet vzoriek, target_size)
    '''
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(datasets) - target_size

    for i in range(start_index, end_index):
        indicies = range(i-history_size, i)
        # reshape data z (history_size, ) na (history_size, 1)
        data.append(np.reshape(datasets[indicies], (history_size, 1)))
        labels.append(datasets[i + target_size])

    return np.array(data), np.array(labels)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    '''
    Parameters
    ----------
    dataset : Data, ktore budu sluzit ako vzorka
    target : Data, ktore budeme chciet predikovat
    start_index : Delenie na testovaciu, validacnu a trenovaciu sadu ->>> zac
    end_index : Delenie na testovaciu, validacnu a trenovaciu sadu ->>> kon
    history_size : velkost vzorky
    target_size : kolko dat dopredu chceme predikovat
    step : ci berieme vsetky data, alebo len nejaku podmnozinu z history_size,
        ak 1 tak berieme vsetky
    single_step : ci chceme predikovat iba jedno dato dopredu, alebo viac

    Returns
    -------
    vracia vzorku a prislusny target
    '''
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indicies = range(i - history_size, i, step) 
        data.append(dataset[indicies])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

# vytvorenie technickych indikatorov
def create_technical_indicators(df, indicators=['MA_4'], only_close_colum=False):
    if only_close_colum:
        if isinstance(df, pd.DataFrame):
            close = np.array(df['Close']).reshape(len(df), )
    else:
        if isinstance(df, pd.DataFrame):
            close = np.array(df['Close']).reshape(len(df), )
            # OPEN = np.array(df['Open']).reshape(len(df), )
            high = np.array(df['High']).reshape(len(df), )
            low = np.array(df['Low']).reshape(len(df), )

    df_2_return = pd.DataFrame()
    for indicator in indicators:
        if indicator is None:
            pass
        
        else:
            if indicator.split('_')[0].strip() == 'MA':
                smavg = talib.MA(close, timeperiod=int(indicator.split('_')[1]))
                df_2_return[indicator] = smavg
            
            if indicator.split('_')[0].strip() == 'BBANDS':
                upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=int(indicator.split('_')[1]))
                df_2_return['lowerband'] = lowerband
                df_2_return[indicator] = middleband
                df_2_return['upperband'] = upperband
                
            if indicator.split('_')[0].strip() == 'MACD':
                indicator = indicator + '_12' + '_26' + '_9'
                _, macdsignal, macdhist = talib.MACD(close,
                                                     fastperiod=int(indicator.split('_')[1]),
                                                     slowperiod=int(indicator.split('_')[2]),
                                                     signalperiod=int(indicator.split('_')[3]))
                df_2_return['macdsignal'] = macdsignal
                df_2_return['macdhist'] = macdhist
            
            if indicator.split('_')[0].strip() == 'RSI':
                real_RSI = talib.RSI(close, timeperiod=int(indicator.split('_')[1]))
                df_2_return[indicator] = real_RSI
            
            if indicator.split('_')[0].strip() == 'CMO':
                real_CMO = talib.CMO(close, timeperiod=int(indicator.split('_')[1]))
                df_2_return[indicator] = real_CMO
            
            if indicator.split('_')[0].strip() == 'ADX':
                if only_close_colum:
                    pass
                else:
                    real_ADX = talib.ADX(high, low, close, timeperiod=int(indicator.split('_')[1]))
                    df_2_return[indicator] = real_ADX
                    
            if indicator.split('_')[0].strip() == 'STOCH':
                if only_close_colum:
                    pass
                else:
                    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                    df_2_return[indicator] = slowd - slowk
    return(df_2_return)

# funkcia vrati pocet neuronov v kazdej vrstve, 2/3 celkovy pocet nueronov zo vstupnych
def neurons_in_layer_two_thirds(first_layer_size, no_of_layers, last_layer_size=1):
    total_neurons = (2/3)*first_layer_size
    neurons_in_layers = np.zeros(no_of_layers)
    if len(neurons_in_layers) > 1:
        neurons_in_layers[0] = total_neurons/2
        
        for i in range(1, no_of_layers):
            neurons_in_layers[i] = neurons_in_layers[i-1]/2
    else:
        neurons_in_layers[0] = total_neurons
        
    result = np.ndarray.astype(np.round(neurons_in_layers), dtype='int')
    
    return(result[result > last_layer_size])

# funkcia vrati pocet neuronov v kazdej vrstve, 1/2 celkovy pocet nueronov zo vstupnych
def neurons_in_layer_half_decreasing(first_layer_size, no_of_layers, last_layer_size=1):
    neurons_in_layers = np.ones(no_of_layers)*first_layer_size
    coeff = np.power(2*np.ones(no_of_layers), np.arange(1, (no_of_layers+1)))
    
    result = np.ndarray.astype(np.round(neurons_in_layers /coeff ), dtype='int')    
    
    
    return(result[result > last_layer_size])

# vypocet kvantilu vynosov
def return_quantile_of_returns(df, quantile=.2, target_column_name=['Close']):
    abs_returns = np.abs(np.array(df[target_column_name][1:]) / np.array(df[target_column_name][:-1]) - 1)
    
    return(np.quantile(abs_returns, q=quantile))