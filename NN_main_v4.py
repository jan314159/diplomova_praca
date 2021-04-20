# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 18:47:15 2020

@author: janpi

Kod na modelovanie cien
"""

import tensorflow as tf
import numpy as np
import os
import pandas as pd

ticker_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/Data'
code_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/kody'
models_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/modely'
data_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/Data/intra'
pics_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/obrázky'

os.chdir(code_source_directory)

import help_fun as hf
import matplotlib.pyplot as plt
import NN_model_v2 as nnmod
import PL_strategy_v2 as PL

# =============================================================================
# Nastavenie parametrov
# =============================================================================

train_split_percentage = .7
val_split_percentage = (1 - train_split_percentage) / 2

normalization_method = 'min_max'
data_logaritmization = False
correct_normalization = False

past_window_examples = 30
future_window_predictions = 1
shift = 1
target_column_name = ['Close']
BATCH = 32


tickers_list = 'tickers.txt'

# =============================================================================
# =============================================================================
os.chdir(models_source_directory)
writer_best_model = pd.ExcelWriter('NN_Price_results.xlsx', engine='xlsxwriter')
writer_all_results = pd.ExcelWriter('NN_all_Price_results.xlsx', engine='xlsxwriter')

os.chdir(ticker_source_directory)

with open(tickers_list, 'r') as t:
    tickers = t.readlines()

# nastavenie priecinku, v ktorom su data

# WindowGenerator upraveny z: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb
class WindowGenerator():
    # nastavenie defaultnych parametrov, s ktorymi dalej pracujeme
    def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df, BATCH=32, label_columns=None):
        # natriedenie a ulozenie datasetov do  
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.BATCH = BATCH # kolko dat chceme mat v jednom kosi dat, ktory vyjde z generatora
        
        
        self.label_columns = label_columns # nastavenie stlpcov, ktore budeme predikovat
        
        if label_columns is not None:
            # vytvori dvojice {'nazov predikovaneho stlpca': jeho poradie v tych, ktore chceme predikovat (od 0.)}
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        
        # vytvroi dvojice {'Nazov stlpca': poradie v dataframe}, dales sa pouzije na splitovanie
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
        
        # nastavenie parametrov
        self.input_width = input_width # kolko dni do minulosti sa divame
        self.label_width = label_width # kolko dni do buducnosti sa divame 
        self.shift = shift 
        
        # celkova velkost okna, predkcie + posun
        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width) # vytvori slace o velkosti input_width, napr ak berieme 10 dni tak slice o dlzke 10
        self.input_indices = np.arange(self.total_window_size)[self.input_slice] # vytvori postupnos od 0 do velkosti okna vstupnych parametrov

        self.label_start = self.total_window_size - self.label_width # pociatocny index, kde zacinaju predikovane hodnoty
        self.labels_slice = slice(self.label_start, None) # slice o velkosti labelu
        # arrange vytvori postupnost cisel 0, 1, 2..., a nasledne sa zoberie slice o velkosti aku, predikujeme 
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice] # urci index labelu
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    # rozdelenie na sady
    def split_window(self, features):
        
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
    
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    # vytvorenie testovacej sady
    def make_test_set(self):
        test_input_data = []
        test_labels = []
        
        test_data = np.array(self.test_df)
        test_targets = np.array(self.test_df[self.label_columns])
        
        end_index = len(self.test_df) - self.label_width
        
        for i in range(self.input_width, end_index):
            indicies = range(i - self.input_width, i, self.shift)
            test_input_data.append(test_data[indicies])
            
            test_labels.append(test_targets[i:(i+self.label_width)])
        
        test_labels = np.array(test_labels)
        test_labels = test_labels.reshape(test_labels.shape[0], test_labels.shape[2])
        test_input_data = np.array(test_input_data)
        
        return test_input_data, test_labels
    
    # vytvorenie validacnej sady
    def make_val_set(self):
        val_input_data = []
        val_labels = []
        
        val_data = np.array(self.val_df)
        val_targets = np.array(self.val_df[self.label_columns])
        
        end_index = len(self.val_df) - self.label_width
        
        for i in range(self.input_width, end_index):
            indicies = range(i - self.input_width, i, self.shift)
            val_input_data.append(val_data[indicies])
            
            val_labels.append(val_targets[i:(i+self.label_width)])
        
        val_labels = np.array(val_labels)
        val_labels = val_labels.reshape(val_labels.shape[0], val_labels.shape[2])
        val_input_data = np.array(val_input_data, dtype='float64')
        
        return val_input_data, val_labels    

    # generator dat pre trenovanie
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.BATCH,)

        ds = ds.map(self.split_window)

        return ds
    
    # vytvorenie generatorov pre jednotlive sady
    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


# funkcia, ktora natrenuje modely
def train_model(df, ticker, indicators=None, correct_normalization=False,
                data_logaritmization=False, normalization_method='min_max',
                TRAIN_SPLIT=None, VAL_SPLIT=None, past_windows_min=10,
                past_windows_max=50, past_window_step=10, future_window_predictions=1,
                layer_no_min=0, layer_no_max=6, patience=2, BATCH=32, max_epochs=30,
                random_drop_net_config_treshhold=None, target_column_name=['Close'],
                final_model_destination='C:/Users/janpi/Desktop/Diplomka/new/modely',
                treshold_in_PL=0.03):
    '''
    funkcia natrenuje a vyberie najlepsi model zo zadanych parametrov     

    Parameters
    ----------
    df : vstupny dataset, musi obsahovat stlpec s target_column_name
    ticker : ticker danej akcie, pre ktoru trenujeme model
    indicators : Indikatory, pre ktore chceme model testovat, ak None,
        zoberu sa vsetky, ak nechceme ziadny tak indicators=['None'] 
        The default is None.
    correct_normalization : Ak True, noralizujeme dataset pomocou parametrov 
        trenovacej sady, ak false, kazdy pomocou svojich parametrov. 
        The default is False.
    data_logaritmization : Ak True, pracujeme s logaritmickymi datami.
        The default is False.
    normalization_method : Druh normalizacie, bud 'min_max' alebo 'standardization'
        The default is 'min_max'.
    TRAIN_SPLIT : Do ktoreho data su trenovacie. The default is None.
    VAL_SPLIT : Do ktoreho data su validacne. The default is None.
    past_windows_min : Minimalne kolko dat do minulosti sa divame.
        The default is 10.
    past_windows_max : Maximalne kolko dat do minulosti sa divame. 
        The default is 50.
    past_window_step : Po kolkych datach sa posuvame. The default is 10.
    future_window_predictions : Kolko obdobi do predu chceme predikovat.
        The default is 1.
    layer_no_min : Min index siete. The default is 0.
    layer_no_max : Max index siete. The default is 6.
    patience : Po kolkych epochach ukoncime optimalizaciu, ak sa nezlepsuje
        validacna strata. The default is 2.
    BATCH : Velkost batchovania. The default is 32.
    max_epochs : Maximalne kolko epoch dovolime. The default is 30.
    random_drop_net_config_treshhold : Max pravdepodobnost s akou chceme
        vynechat dane nastavenie siete. Ak je None, tak chceme vsetky a nic
        nevynechavame. The default is None.
    target_column_name : Co chceme predikovat. The default is ['Close'].

    Returns
    -------
    Funkcia vracia vysledky vsetkych sieti a parametre najlepsej siete a
    uklada najlepsi model. 

    '''
    # zadefinovanie pomocnych premennych
    capital_old = 0
    sharpe_old = 0
    
    check_parameter = 0
    
    model_params = []
    final_indicators = [[]]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    
    os.chdir(final_model_destination)
    no = 0
    
    # prejdeme cez vsetky hyperparametre
    if indicators is None:
        indicators = [[None], ['MA_4', 'MA_9', 'MA_18'], ['BBANDS_18'],
                      ['MACD'], ['RSI_14'], ['ADX_14'], ['STOCH']]
        
    
    # vytvorenie datasetu s pridanym indikatorom
    for indicator in indicators:
        model_dataset = pd.concat([
            df, hf.create_technical_indicators(df, indicators=indicator)], axis=1)

        model_dataset = model_dataset.dropna()
        
        # ak logaritmujeme data
        if data_logaritmization:
            model_dataset = np.log(model_dataset)
        
        # znormalizovanie dat
        if normalization_method == 'min_max':
            
            if correct_normalization:
                model_dataset, data_min, data_max = hf.normalize_data(
                    model_dataset, split=TRAIN_SPLIT, normalization='min_max')
            else:
                model_dataset, data_train_min, data_train_max,\
                    data_val_min, data_val_max,\
                        data_test_min, data_test_max = hf.normalize_each_subset_alone(
                            model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
                        
        elif normalization_method == 'standardization':
            
            if correct_normalization:
                model_dataset, data_mean, data_std = hf.normalize_data(
                    model_dataset, split=TRAIN_SPLIT, normalization='standardization')
            else:
                model_dataset, data_train_mean, data_train_std, \
                    data_val_mean, data_val_std, data_test_mean, \
                        data_test_std = hf.normalize_each_subset_alone(
                            model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                            normalization='standardization')
        
        
        # zbavime sa Nan-ov, ak nejake vznikli
        model_dataset = model_dataset.dropna() 
     
        # vytvorenie datasetu podla vstupnych dat
        for past_window_examples in range(past_windows_min, past_windows_max+1, past_window_step):
            train_df = model_dataset[:TRAIN_SPLIT]
            val_df = model_dataset[TRAIN_SPLIT:VAL_SPLIT]
            test_df = model_dataset[VAL_SPLIT:]
            
            data_generator = WindowGenerator(
                input_width=past_window_examples, label_width=future_window_predictions,
                shift=shift,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                label_columns=target_column_name,
                BATCH=BATCH) 
            
            val_input_data, val_label_data = data_generator.make_val_set()
            
            if correct_normalization:
                val_labels_denormalized = hf.denormalize_data(
                    val_label_data, logaritmization=data_logaritmization,
                    data_max=data_max[target_column_name[0]],
                    data_min=data_min[target_column_name[0]])
            else:
                val_labels_denormalized = hf.denormalize_data(
                    val_label_data, logaritmization=data_logaritmization,
                    data_max=data_val_max[target_column_name[0]],
                    data_min=data_val_min[target_column_name[0]])
            
            # urcenie siete
            for layer_number in range(layer_no_min, layer_no_max+1):
                first_layer_size = data_generator.train.element_spec[0].shape[1] * data_generator.train.element_spec[0].shape[2]
                no_of_layers = nnmod.return_length_of_hidden_layers(layer_number)
                
                
                neurons_2_3 = hf.neurons_in_layer_two_thirds(
                    first_layer_size=first_layer_size, no_of_layers=no_of_layers,
                    last_layer_size=1)
                
                
                # neurons_half_decrease = hf.neurons_in_layer_half_decreasing(
                    # first_layer_size=first_layer_size, no_of_layers=no_of_layers,
                    # last_layer_size=1)
                
                # for NEURONS in [neurons_2_3, neurons_half_decrease]:
                NEURONS = neurons_2_3
                
                # nattrenovanie a vypocet predikovanych hodnot pre validacnu sadu
                if len(NEURONS) == no_of_layers:
                    if (random_drop_net_config_treshhold is None or 
                        np.random.rand(1) > random_drop_net_config_treshhold):
                        
                        
                        model = nnmod.create_RNN_model(neurons_in_layers=NEURONS,
                                                       model_index=layer_number)
                
                        history = model.fit(data_generator.train, epochs=max_epochs,
                                            validation_data=data_generator.val, 
                                            callbacks=[early_stopping])
                
                        val_predictions = model.predict(val_input_data)
                        
                        if correct_normalization:
                            val_predictions_denormalized = hf.denormalize_data(
                                val_predictions, logaritmization=data_logaritmization,
                                data_max=data_max[target_column_name[0]],
                                data_min=data_min[target_column_name[0]])
                        else:
                            val_predictions_denormalized = hf.denormalize_data(
                                val_predictions, logaritmization=data_logaritmization,
                                data_max=data_val_max[target_column_name[0]],
                                data_min=data_val_min[target_column_name[0]])
                            
                         
                        
                        capital, sharpe,_,_ = PL.PL_backtesting_v2(
                            val_labels_denormalized, val_predictions_denormalized,
                            threshold=treshold_in_PL)
                        #=====================================================
                        # buy_sell = np.array([buy_sell_indicator.reshape(len(buy_sell_indicator), ),
                        #                      val_labels_denormalized[:-1].reshape(len(buy_sell_indicator), )]).T
                        
                        
                        # plt.rcParams["figure.figsize"] = (8, 4.8)
                        # plt.plot(val_labels_denormalized, color='black', label='skutocne ceny')
                        # plt.plot(val_predictions_denormalized, color='red', label='predikcie')
                        # plt.scatter(np.where(buy_sell[:, 0] == 1), buy_sell[np.where(buy_sell[:, 0] == 1)][:,1], label='long pozicia', color='green', s=15, marker='^')
                        # plt.scatter(np.where(buy_sell[:, 0] == -1), buy_sell[np.where(buy_sell[:, 0] == -1)][:,1], label='short pozicia', color='red', s=15, marker='v')
                        # plt.scatter(np.where(buy_sell[:, 0] == 0), buy_sell[np.where(buy_sell[:, 0] == 0)][:,1], label='n/a', color='blue', s=15, marker='>')
                        # plt.xlabel('tick')
                        # plt.ylabel('cena')
                        # plt.title(no)
                        # plt.legend()
                        # plt.show()
                        #=====================================================
                        
                        
                        
                        model_params.append([
                            layer_number, NEURONS, past_window_examples,
                            model_dataset.columns.tolist(), capital[0], sharpe[0],
                            history.history['val_loss'][-1],
                            history.history['val_mean_squared_error'][-1]])
                        
                        print('===========')
                        print(no)
                        print(indicator)
                        print(past_window_examples)
                        print(layer_number)
                        print('===========')
                        no += 1
                        if capital > capital_old and sharpe > sharpe_old:
                    
                            final_model = model
                            final_indicator = indicator.copy()
                            
                            final_indicators[check_parameter] = final_indicator
                            
                            final_layer_number = layer_number
                            final_neurons_in_layer = NEURONS
                            
                            final_val_loss = history.history['val_loss'][-1]
                            final_val_mean_squared_error = history.history['val_mean_squared_error'][-1]
                            
                            capital_old = capital[0].copy()
                            sharpe_old = sharpe[0].copy()
                            
                            final_past_window_examples = past_window_examples
                            
                        del(model)
            
                      
                        
            
    # uprava indikatorov
    
    if final_indicator == [None]:
        indicators = []          
    
    else:
        indicators.pop(indicators.index(final_indicator))
        indicators.pop(indicators.index([None]))
        df = pd.concat([
            df, hf.create_technical_indicators(df, indicators=final_indicator)], axis=1)
        
        final_indicators.append([])
        
    while len(indicators) > 0: 
        check_parameter += 1
        
        
        for indicator in indicators:
        
            
            
            model_dataset = pd.concat(
                [df, hf.create_technical_indicators(df, indicators=indicator)], axis=1)

            model_dataset = model_dataset.dropna()
            
            if data_logaritmization:
                model_dataset = np.log(model_dataset)
    
            if normalization_method == 'min_max':
                
                if correct_normalization:
                    model_dataset, data_min, data_max = hf.normalize_data(
                        model_dataset, split=TRAIN_SPLIT, normalization='min_max')
                else:
                    model_dataset, data_train_min, data_train_max,\
                        data_val_min, data_val_max,\
                            data_test_min, data_test_max = hf.normalize_each_subset_alone(
                                model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
                            
            elif normalization_method == 'standardization':
                
                if correct_normalization:
                    model_dataset, data_mean, data_std = hf.normalize_data(
                        model_dataset, split=TRAIN_SPLIT, normalization='standardization')
                else:
                    model_dataset, data_train_mean, data_train_std, \
                        data_val_mean, data_val_std, data_test_mean, \
                            data_test_std = hf.normalize_each_subset_alone(
                                model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                                normalization='standardization')
            
            

            model_dataset = model_dataset.dropna() 
            
            train_df = model_dataset[:TRAIN_SPLIT]
            val_df = model_dataset[TRAIN_SPLIT:VAL_SPLIT]
            test_df = model_dataset[VAL_SPLIT:]
            
            data_generator = WindowGenerator(
                input_width=final_past_window_examples,
                label_width=future_window_predictions,
                shift=shift,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                label_columns=target_column_name,
                BATCH=BATCH) 
            
            val_input_data, val_label_data = data_generator.make_val_set()
            if correct_normalization:
                val_labels_denormalized = hf.denormalize_data(
                    val_label_data, logaritmization=data_logaritmization,
                    data_max=data_max[target_column_name[0]],
                    data_min=data_min[target_column_name[0]])
            else:
                val_labels_denormalized = hf.denormalize_data(
                    val_label_data, logaritmization=data_logaritmization,
                    data_max=data_val_max[target_column_name[0]],
                    data_min=data_val_min[target_column_name[0]])
            
            first_layer_size = data_generator.train.element_spec[0].shape[1] * data_generator.train.element_spec[0].shape[2]
            no_of_layers = nnmod.return_length_of_hidden_layers(final_layer_number)
            
            neurons_2_3 = hf.neurons_in_layer_two_thirds(
                        first_layer_size=first_layer_size,
                        no_of_layers=no_of_layers,
                        last_layer_size=1)
                    
                    
            # neurons_half_decrease = hf.neurons_in_layer_half_decreasing(
                # first_layer_size=first_layer_size, no_of_layers=no_of_layers,
                # last_layer_size=1)
            
            # for NEURONS in [neurons_2_3, neurons_half_decrease]:
            NEURONS = neurons_2_3
            
            model = nnmod.create_RNN_model(neurons_in_layers=final_neurons_in_layer,
                                           model_index=final_layer_number)
                    
            history = model.fit(data_generator.train, epochs=max_epochs,
                                validation_data=data_generator.val, 
                                callbacks=[early_stopping])
    
            val_predictions = model.predict(tf.convert_to_tensor(val_input_data))
    
            if correct_normalization:
                val_predictions_denormalized = hf.denormalize_data(
                    val_predictions, logaritmization=data_logaritmization,
                    data_max=data_max[target_column_name[0]],
                    data_min=data_min[target_column_name[0]])
            else:
                val_predictions_denormalized = hf.denormalize_data(
                    val_predictions, logaritmization=data_logaritmization,
                    data_max=data_val_max[target_column_name[0]],
                    data_min=data_val_min[target_column_name[0]])
            
            capital, sharpe, _, _ = PL.PL_backtesting_v2(
                val_labels_denormalized, val_predictions_denormalized,
                threshold=treshold_in_PL)
            
            model_params.append([
                final_layer_number, final_neurons_in_layer,
                final_past_window_examples,
                model_dataset.columns.tolist(), capital[0], sharpe[0],
                history.history['val_loss'][-1],
                history.history['val_mean_squared_error'][-1]])
            
            no += 1
            print('===========')
            print(no)
            print(indicator)
            print(past_window_examples)
            print(layer_number)
            print('===========')
            if capital > capital_old and sharpe > sharpe_old:
        
                final_model = model
                
                final_indicator = indicator.copy()
                
                final_indicators[check_parameter] = final_indicator
                
                
                final_val_loss = history.history['val_loss'][-1]
                final_val_mean_squared_error = history.history['val_mean_squared_error'][-1]
                
                capital_old = capital[0].copy()
                sharpe_old = sharpe[0].copy()
                
                
            del(model)
        
        
        if final_indicator in indicators:
            
            indicators.pop(indicators.index(final_indicator))
            df = pd.concat([
                df, hf.create_technical_indicators(df, indicators=final_indicator)], axis=1)
            
            final_indicators.append([])
        else:
            indicators = []
        
        
        
        
    
    if final_indicators[-1] == []:
        final_indicators.pop(-1)
    
    final_model.save(ticker+'.h5')
    del(final_model)
    
    return(model_params, sum(final_indicators, []), final_layer_number,
           final_neurons_in_layer, final_val_loss, final_val_mean_squared_error,
           final_past_window_examples, capital_old, sharpe_old)


# funkcia, ktora spocita vysledky na testovacej sade a nakresli obrazok
def test_final_model(df, ticker, indicators, final_past_window_examples,
                     correct_normalization=False, data_logaritmization=False,
                     normalization_method='min_max', TRAIN_SPLIT=None,
                     VAL_SPLIT=None, BATCH=32, target_column_name=['Close'],
                     final_model_destination='C:/Users/janpi/Desktop/Diplomka/new/modely',
                     pics_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/obrázky',
                     treshold_in_PL=0.03):
    capital = 0
    sharpe = 0
    
    os.chdir(final_model_destination)
    
    final_model_dataset = pd.concat(
        [df,  hf.create_technical_indicators(df, indicators=indicators)], axis=1)
    
    final_model_dataset = final_model_dataset.dropna()
    if data_logaritmization:
                final_model_dataset = np.log(final_model_dataset)
    
    if normalization_method == 'min_max':
                
        if correct_normalization:
            final_model_dataset, data_min, data_max = hf.normalize_data(
                final_model_dataset, split=TRAIN_SPLIT, normalization='min_max')
        else:
            final_model_dataset, data_train_min, data_train_max,\
                data_val_min, data_val_max,\
                    data_test_min, data_test_max = hf.normalize_each_subset_alone(
                        final_model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
                
    elif normalization_method == 'standardization':
        
        if correct_normalization:
            final_model_dataset, data_mean, data_std = hf.normalize_data(
                final_model_dataset, split=TRAIN_SPLIT, normalization='standardization')
        else:
            final_model_dataset, data_train_mean, data_train_std, \
                data_val_mean, data_val_std, data_test_mean, \
                    data_test_std = hf.normalize_each_subset_alone(
                        final_model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
                        normalization='standardization')    
                    
                    
    
    final_model_dataset = final_model_dataset.dropna()
                    

    train_df = final_model_dataset[:TRAIN_SPLIT]
    val_df = final_model_dataset[TRAIN_SPLIT:VAL_SPLIT]
    test_df = final_model_dataset[VAL_SPLIT:]
    
    data_generator = WindowGenerator(
                    input_width=final_past_window_examples,
                    label_width=future_window_predictions,
                    shift=shift,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    label_columns=target_column_name,
                    BATCH=BATCH)
    
    test_input_data, test_label_data = data_generator.make_test_set()
    
    if correct_normalization:
        test_labels_denormalized = hf.denormalize_data(
            test_label_data, logaritmization=data_logaritmization,
            data_max=data_max[target_column_name[0]],
            data_min=data_min[target_column_name[0]])
    else:
        test_labels_denormalized = hf.denormalize_data(
            test_label_data, logaritmization=data_logaritmization,
            data_max=data_test_max[target_column_name[0]],
            data_min=data_test_min[target_column_name[0]])
    
    
    model = tf.keras.models.load_model(ticker+'.h5')
    
    test_predictions = model.predict(tf.convert_to_tensor(test_input_data))
    
    if correct_normalization:
        test_predictions_denormalized = hf.denormalize_data(
            test_predictions, logaritmization=data_logaritmization,
            data_max=data_max[target_column_name[0]],
            data_min=data_min[target_column_name[0]])
    else:
        test_predictions_denormalized = hf.denormalize_data(
            test_predictions, logaritmization=data_logaritmization,
            data_max=data_test_max[target_column_name[0]],
            data_min=data_test_min[target_column_name[0]])
    
    
    
    capital, sharpe, buy_sell_indicator, _ = PL.PL_backtesting_v2(test_labels_denormalized,
                                                               test_predictions_denormalized,
                                                               threshold=treshold_in_PL)
    evaluate_model = model.evaluate(test_input_data, test_label_data)
    
    buy_sell = np.array([buy_sell_indicator.reshape(len(buy_sell_indicator), ),
                  test_labels_denormalized[:-1].reshape(len(buy_sell_indicator), )]).T
        
    os.chdir(pics_source_directory)
    plt.rcParams["figure.figsize"] = (8, 4.8)
    plt.plot(test_labels_denormalized, color='black', label='skutocne ceny')
    plt.plot(test_predictions_denormalized, color='red', label='predikcie')
    plt.scatter(np.where(buy_sell[:, 0] == 1), buy_sell[np.where(buy_sell[:, 0] == 1)][:,1], label='long pozicia', color='green', s=15, marker='^')
    plt.scatter(np.where(buy_sell[:, 0] == -1), buy_sell[np.where(buy_sell[:, 0] == -1)][:,1], label='short pozicia', color='red', s=15, marker='v')
    plt.scatter(np.where(buy_sell[:, 0] == 0), buy_sell[np.where(buy_sell[:, 0] == 0)][:,1], label='n/a', color='blue', s=15, marker='>')
    plt.xlabel('tick')
    plt.ylabel('cena')
    plt.title(ticker)
    plt.legend()
    plt.savefig(ticker+'.png', dpi=1000)
    plt.show()
    
    
    
    del(model)
    # [test capital, test sharpe, test mae, test mse]
    results = [evaluate_model[0], evaluate_model[2], capital[0], sharpe[0]]
    return(results)


# nazvy stlpcov vo vysledkoch
columns_all_results = ['layer no', 'neurons in layer', 'past window', 'indicators_used',
        'val capital', 'val sharpe', 'val loss', 'val MSE']

columns_final_results = ['ticer', 'layer no', 'neurons in layer', 'val loss', 'val mse',
                         'val capital', 'val sharpe', 'indicators', 'past windows',
                         'test loss', 'test mse', 'test PL', 'test sharpe']
row = 1
pd.DataFrame([] ,columns=columns_final_results).to_excel(writer_best_model, index=False, header=True)
for tick in tickers:
    os.chdir(data_source_directory)
    df = pd.read_excel(tick.rstrip('\n')+'.xlsx')

    # denne data    

    # df = df[['adj_close',
    #           'adj_high',
    #           'adj_low',
    #           'adj_open']]

    # df = df.rename(columns={'adj_close' : 'Close',
    #                         'adj_high' : 'High',
    #                         'adj_low' : 'Low',
    #                         'adj_open' : 'Open'})
    
    # df.dropna(inplace=True)
    # df.reset_index(drop=True, inplace=True)
    
    # hodinove data 
    
    df = df[['close',
              'high',
              'low',
              'open']]

    df = df.rename(columns={'close' : 'Close',
                            'high' : 'High',
                            'low' : 'Low',
                            'open' : 'Open'})
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    
    df_length = len(df)

    # nastavenie priecinku s kodmi
    os.chdir(code_source_directory)
    
    # pocet dat v jednotlivych setoch
    TRAIN_SPLIT = int(np.round(train_split_percentage * df_length))
    VAL_SPLIT = int(TRAIN_SPLIT + np.round(val_split_percentage * df_length))
    
    treshold_in_PL = hf.return_quantile_of_returns(df[:TRAIN_SPLIT], quantile=.2,
                                                   target_column_name=target_column_name)
    
    indicators = [[None], ['MA_4', 'MA_9', 'MA_18'], ['BBANDS_18'],
                    ['MACD'], ['RSI_14'], ['ADX_14'], ['STOCH']]
    
    
    # natrenovanie modela
    model_params, final_indicators, final_layer_number, final_neurons_in_layer, \
        final_val_loss, final_val_mean_squared_error, \
            final_past_window_examples, val_capital, val_sharpe = train_model(
                df, ticker=tick.rstrip('\n'), indicators=indicators,
                correct_normalization=correct_normalization,
                data_logaritmization=data_logaritmization, 
                normalization_method=normalization_method,
                TRAIN_SPLIT=TRAIN_SPLIT, VAL_SPLIT=VAL_SPLIT,
                past_windows_min=10, past_windows_max=50, past_window_step=10,
                future_window_predictions=future_window_predictions,
                layer_no_min=0, layer_no_max=8,
                patience=2, BATCH=BATCH, max_epochs=30,
                random_drop_net_config_treshhold=None,
                target_column_name=target_column_name,
                final_model_destination=models_source_directory,
                treshold_in_PL=treshold_in_PL)
    
    pd.DataFrame(model_params, columns=columns_all_results).to_excel(writer_all_results, sheet_name=tick.rstrip('\n'))
    
    # otestovanie finalneho modela
    final_results = test_final_model(
        df, ticker=tick.rstrip('\n'), indicators=final_indicators,
        final_past_window_examples=final_past_window_examples,
        correct_normalization=correct_normalization,
        data_logaritmization=data_logaritmization, 
        normalization_method=normalization_method,
        TRAIN_SPLIT=TRAIN_SPLIT, VAL_SPLIT=VAL_SPLIT,
        BATCH=32, final_model_destination=models_source_directory,
        treshold_in_PL=treshold_in_PL)
    
    best_model_results = np.array([
        tick.rstrip('\n'), final_layer_number, final_neurons_in_layer,
        final_val_loss, final_val_mean_squared_error, val_capital, val_sharpe,
        final_indicators, final_past_window_examples, final_results[0],
        final_results[1], final_results[2], final_results[3]], dtype='object_')
    
    pd.DataFrame(best_model_results.reshape(1, 13)).to_excel(writer_best_model, index=False, header=False, startrow=row)
    row += 1

# ulozenie vysledkov
os.chdir(models_source_directory)    
writer_all_results.save()
writer_best_model.save()

