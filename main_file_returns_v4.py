# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:12:44 2020

@author: janpi

Kod na modelovanie vynosov
"""


import tensorflow as tf
import numpy as np
import os
import pandas as pd
from sklearn.metrics import r2_score

ticker_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/Data'
data_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/Data/intra'
code_source_directory = 'C:/Users/janpi/Desktop/Diplomka/new/kody'
models_source_directory = 'C:/Users/janpi/Desktop/Diplomka/returns/modely'
pics_source_directory = 'C:/Users/janpi/Desktop/Diplomka/returns/obrazky'

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
correct_normalization = True

# past_window_examples = 30
future_window_predictions = 1
shift = 1
target_column_name = ['Close']
BATCH = 32

multiplication_constant = 100

tickers_list = 'tickers.txt'
# =============================================================================
# =============================================================================
os.chdir(models_source_directory)
writer_best_model = pd.ExcelWriter('NN_returns_Price_results.xlsx', engine='xlsxwriter')
writer_all_results = pd.ExcelWriter('NN_returns_all_Price_results.xlsx', engine='xlsxwriter')

os.chdir(ticker_source_directory)
with open(tickers_list, 'r') as t:
    tickers = t.readlines()


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
        self.shift = shift # vynechanie nasobkov z labelov napr, ak by sme chceli 1 label a posun by bol 30 tak berieme az o 30 indexov
            # shift 0 nedava zmysel
        
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
    
    def split_window(self, features):
        
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
    
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
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
        val_input_data = np.array(val_input_data)
        
        return val_input_data, val_labels    
    
    def make_train_set(self):
        train_input_data = []
        train_labels = []
        
        train_data = np.array(self.train_df)
        train_targets = np.array(self.train_df[self.label_columns])
        
        end_index = len(self.train_df) - self.label_width
        
        for i in range(self.input_width, end_index):
            indicies = range(i - self.input_width, i, self.shift)
            train_input_data.append(train_data[indicies])
            
            train_labels.append(train_targets[i:(i+self.label_width)])
        
        train_labels = np.array(train_labels)
        train_labels = train_labels.reshape(train_labels.shape[0], train_labels.shape[2])
        train_input_data = np.array(train_input_data)
        
        return train_input_data, train_labels    
    
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
def train_model(df, true_prices, ticker, indicators=None,
                TRAIN_SPLIT=None, VAL_SPLIT=None, past_windows_min=10,
                past_windows_max=50, past_window_step=10, future_window_predictions=1,
                layer_no_min=0, layer_no_max=6, patience=2, BATCH=32, max_epochs=30,
                random_drop_net_config_treshhold=None, target_column_name=['Close'],
                final_model_destination='C:/Users/janpi/Desktop/Diplomka/new/modely',
                treshold_in_PL=0.03, multiplication_constant=10):
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
    capital_old = 0
    sharpe_old = 0
    
    final_val_loss = 1000
    final_val_mean_squared_error = 1000
    
    check_parameter = 0
    no = 0
    final_indicator = [None]
    
    model_params = []
    final_indicators = [[]]
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    
    os.chdir(final_model_destination)
    
    if indicators is None:
        indicators = [[None], ['MA_4', 'MA_9', 'MA_18'], ['BBANDS_18'],
                      ['MACD'], ['RSI_14'], ['ADX_14'], ['STOCH']]
        
    
        
    for indicator in indicators:
        # model_dataset = pd.concat([df], axis=1)
        
        model_dataset = pd.concat([
            df, hf.create_technical_indicators(df, indicators=indicator)], axis=1)
        model_dataset = model_dataset.dropna()    

        # if normalization_method == 'min_max':
            
        #     if correct_normalization:
        #         model_dataset, data_min, data_max = hf.normalize_data(
        #             model_dataset, split=TRAIN_SPLIT, normalization='min_max')
        #     else:
        #         model_dataset, data_train_min, data_train_max,\
        #             data_val_min, data_val_max,\
        #                 data_test_min, data_test_max = hf.normalize_each_subset_alone(
        #                     model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
                        
        # elif normalization_method == 'standardization':
            
        #     if correct_normalization:
        #         model_dataset, data_mean, data_std = hf.normalize_data(
        #             model_dataset, split=TRAIN_SPLIT, normalization='standardization')
        #     else:
        #         model_dataset, data_train_mean, data_train_std, \
        #             data_val_mean, data_val_std, data_test_mean, \
        #                 data_test_std = hf.normalize_each_subset_alone(
        #                     model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
        #                     normalization='standardization')
        
        
        # model_dataset = model_dataset.dropna() 
        
        
        true_prices = true_prices[-len(model_dataset):]
        model_dataset = multiplication_constant*model_dataset
        
        for past_window_examples in range(past_windows_min, past_windows_max+1, past_window_step):
            train_df = model_dataset[:TRAIN_SPLIT]
            val_df = model_dataset[TRAIN_SPLIT:VAL_SPLIT]
            test_df = model_dataset[VAL_SPLIT:]
            
            data_generator = WindowGenerator(
                input_width=past_window_examples,
                label_width=future_window_predictions,
                shift=shift,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                label_columns=target_column_name,
                BATCH=BATCH) 
            
            val_input_data, val_label_data = data_generator.make_val_set()
            
            output_bias = tf.keras.initializers.Constant(0)
            for layer_number in range(layer_no_min, layer_no_max+1):
                first_layer_size = data_generator.train.element_spec[0].shape[1] * data_generator.train.element_spec[0].shape[2]
                no_of_layers = nnmod.return_length_of_hidden_layers(layer_number)
                
                
                neurons_2_3 = hf.neurons_in_layer_two_thirds(
                    first_layer_size=first_layer_size, no_of_layers=no_of_layers,
                    last_layer_size=1)
                
                
                # neurons_half_decrease = hf.neurons_in_layer_half_decreasing(
                #     first_layer_size=first_layer_size, no_of_layers=no_of_layers,
                #     last_layer_size=1)
                
                # for NEURONS in [neurons_2_3, neurons_half_decrease]:
                NEURONS = neurons_2_3
                if len(NEURONS) == no_of_layers:
                    if (random_drop_net_config_treshhold is None or 
                        np.random.rand(1) > random_drop_net_config_treshhold):
                        
                        no += 1
                        print('===========')
                        print(no)
                        print(indicator)
                        print(past_window_examples)
                        print(layer_number)
                        print('===========')
                        
                        model = nnmod.create_RNN_model(neurons_in_layers=NEURONS,
                                                       model_index=layer_number,
                                                       loss_function='mae',
                                                       output_activation=None,
                                                       drop=0.0)
                
                        history = model.fit(data_generator.train, epochs=max_epochs,
                                            validation_data=data_generator.val, 
                                            callbacks=[early_stopping])
                
                        val_predictions = model.predict(val_input_data)
                        
                        # plt.plot(val_label_data/multiplication_constant,
                        #          color='blue', label='ture data')
                        # plt.plot(val_predictions/multiplication_constant, 
                        #          color='black', label='predikcie')
                        # # plt.plot(np.zeros(len(val_label_data)), color='red')
                        
                        # plt.legend()
                        # plt.show()
                        
                        # if correct_normalization:
                        #     val_predictions_denormalized = hf.denormalize_data(
                        #         val_predictions, logaritmization=False,
                        #         data_max=data_max[target_column_name[0]],
                        #         data_min=data_min[target_column_name[0]])
                        # else:
                        #     val_predictions_denormalized = hf.denormalize_data(
                        #         val_predictions, logaritmization=False,
                        #         data_max=data_val_max[target_column_name[0]],
                        #         data_min=data_val_min[target_column_name[0]])
                        
                        val_true_prices = np.array(true_prices[(len(train_df)+past_window_examples):(len(train_df)+len(val_df)-future_window_predictions)])

                        # capital, sharpe,_ = PL.PL_backtesting_log_returns(
                        #     val_true_prices.reshape(len(val_true_prices), ),
                        #     val_predictions_denormalized.reshape(len(val_predictions_denormalized), ),
                        #     threshold=treshold_in_PL)
                        
                        capital, sharpe,_ = PL.PL_backtesting_log_returns(
                            val_true_prices.reshape(len(val_true_prices), )/multiplication_constant,
                            val_predictions.reshape(len(val_predictions), )/multiplication_constant,
                            threshold=treshold_in_PL)
                        
                        model_params.append([
                            layer_number, NEURONS, past_window_examples,
                            model_dataset.columns.tolist(), capital, sharpe,
                            history.history['val_loss'][-1],
                            history.history['val_mean_squared_error'][-1]])
                        
                        
                        
                        if ((history.history['val_loss'][-1] < final_val_loss) and 
                            (history.history['val_mean_squared_error'][-1] < final_val_mean_squared_error)):
                    
                            final_model = model
                            final_indicator = indicator.copy()
                            
                            final_indicators[check_parameter] = final_indicator
                            
                            final_layer_number = layer_number
                            final_neurons_in_layer = NEURONS
                            
                            final_val_loss = np.copy(history.history['val_loss'][-1])
                            final_val_mean_squared_error = np.copy(history.history['val_mean_squared_error'][-1])
                            
                            capital_old = capital.copy()
                            sharpe_old = sharpe.copy()
                            
                            final_past_window_examples = past_window_examples
                            
                        del(model)
                            
    # uprava indikatorov
    
    if final_indicator == [None]:
        indicators = []          
    
    else:
        indicators.pop(indicators.index(final_indicator))
        df = pd.concat([
            df, hf.create_technical_indicators(df, indicators=final_indicator)], axis=1)
        
        final_indicators.append([])
    
    while len(indicators) > 0:
        check_parameter += 1
        
        for indicator in indicators:
            model_dataset = pd.concat([
                df, hf.create_technical_indicators(df, indicators=indicator)], axis=1)
            model_dataset = model_dataset.dropna()    
    
            # if normalization_method == 'min_max':
                
            #     if correct_normalization:
            #         model_dataset, data_min, data_max = hf.normalize_data(
            #             model_dataset, split=TRAIN_SPLIT, normalization='min_max')
            #     else:
            #         model_dataset, data_train_min, data_train_max,\
            #             data_val_min, data_val_max,\
            #                 data_test_min, data_test_max = hf.normalize_each_subset_alone(
            #                     model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
                            
            # elif normalization_method == 'standardization':
                
            #     if correct_normalization:
            #         model_dataset, data_mean, data_std = hf.normalize_data(
            #             model_dataset, split=TRAIN_SPLIT, normalization='standardization')
            #     else:
            #         model_dataset, data_train_mean, data_train_std, \
            #             data_val_mean, data_val_std, data_test_mean, \
            #                 data_test_std = hf.normalize_each_subset_alone(
            #                     model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
            #                     normalization='standardization')
            
            
            model_dataset = multiplication_constant*model_dataset
            true_prices = true_prices[-len(model_dataset):]
            
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
            
            no += 1
            print('===========')
            print(no)
            print(indicator)
            print(past_window_examples)
            print(layer_number)
            print('===========')
            
            model = nnmod.create_RNN_model(neurons_in_layers=final_neurons_in_layer,
                                           model_index=final_layer_number,
                                           drop=0.0,
                                           output_activation=None)
                    
            history = model.fit(data_generator.train, epochs=max_epochs,
                                validation_data=data_generator.val, 
                                callbacks=[early_stopping])
            
            val_predictions = model.predict(val_input_data)
            
            # if correct_normalization:
            #     val_predictions_denormalized = hf.denormalize_data(
            #         val_predictions, logaritmization=False,
            #         data_max=data_max[target_column_name[0]],
            #         data_min=data_min[target_column_name[0]])
            # else:
            #     val_predictions_denormalized = hf.denormalize_data(
            #         val_predictions, logaritmization=False,
            #         data_max=data_val_max[target_column_name[0]],
            #         data_min=data_val_min[target_column_name[0]])
            
            val_true_prices = np.array(true_prices[(len(train_df)+final_past_window_examples):(len(train_df)+len(val_df)-future_window_predictions)])
            
            # capital, sharpe,_ = PL.PL_backtesting_log_returns(
            #                 val_true_prices.reshape(len(val_true_prices), ),
            #                 val_predictions_denormalized.reshape(len(val_predictions_denormalized), ),
            #                 threshold=treshold_in_PL)
            
            capital, sharpe,_ = PL.PL_backtesting_log_returns(
                            val_true_prices.reshape(len(val_true_prices), )/multiplication_constant,
                            val_predictions.reshape(len(val_predictions), )/multiplication_constant,
                            threshold=treshold_in_PL)
            
            model_params.append([
                layer_number, NEURONS, past_window_examples,
                model_dataset.columns.tolist(), capital, sharpe,
                history.history['val_loss'][-1],
                history.history['val_mean_squared_error'][-1]])
            
            if ((history.history['val_loss'][-1] < final_val_loss) and 
                (history.history['val_mean_squared_error'][-1] < final_val_mean_squared_error)):
                
                final_model = model
                final_indicator = indicator.copy()
                
                final_indicators[check_parameter] = final_indicator
                
                final_layer_number = layer_number
                final_neurons_in_layer = NEURONS
                
                final_val_loss = history.history['val_loss'][-1]
                final_val_mean_squared_error = history.history['val_mean_squared_error'][-1]
                
                capital_old = capital.copy()
                sharpe_old = sharpe.copy()
                
                final_past_window_examples = past_window_examples
                
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

def test_final_model(df, ticker, true_prices, indicators, final_past_window_examples,
                     correct_normalization=False, data_logaritmization=False,
                     normalization_method='min_max', TRAIN_SPLIT=None,
                     VAL_SPLIT=None, BATCH=32, target_column_name=['Close'],
                     final_model_destination='C:/Users/janpi/Desktop/Diplomka/new/modely',
                     pics_source_directory=pics_source_directory,
                     treshold_in_PL=0.03, multiplication_constant=10):
    capital = 0
    sharpe = 0
    
    os.chdir(final_model_destination)
    
    final_model_dataset = pd.concat([
                df, hf.create_technical_indicators(df, indicators=indicators)], axis=1)
    final_model_dataset = final_model_dataset.dropna()
    final_model_dataset = multiplication_constant*final_model_dataset
    
   
    # if normalization_method == 'min_max':
                
    #     if correct_normalization:
    #         final_model_dataset, data_min, data_max = hf.normalize_data(
    #             final_model_dataset, split=TRAIN_SPLIT, normalization='min_max')
    #     else:
    #         final_model_dataset, data_train_min, data_train_max,\
    #             data_val_min, data_val_max,\
    #                 data_test_min, data_test_max = hf.normalize_each_subset_alone(
    #                     final_model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT)
                
    # elif normalization_method == 'standardization':
        
    #     if correct_normalization:
    #         final_model_dataset, data_mean, data_std = hf.normalize_data(
    #             final_model_dataset, split=TRAIN_SPLIT, normalization='standardization')
    #     else:
    #         final_model_dataset, data_train_mean, data_train_std, \
    #             data_val_mean, data_val_std, data_test_mean, \
    #                 data_test_std = hf.normalize_each_subset_alone(
    #                     final_model_dataset, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT,
    #                     normalization='standardization')    
    
    
    
    true_prices = true_prices[-len(final_model_dataset):]
    
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
    
    # if correct_normalization:
    #     test_label_data_denormalized = hf.denormalize_data(
    #         test_label_data, logaritmization=False,
    #         data_max=data_max[target_column_name[0]],
    #         data_min=data_min[target_column_name[0]])
    # else:
    #     test_label_data_denormalized = hf.denormalize_data(
    #         test_label_data, logaritmization=False,
    #         data_max=data_test_max[target_column_name[0]],
    #         data_min=data_test_min[target_column_name[0]])
    
    model = tf.keras.models.load_model(ticker+'.h5')
    
    test_predictions = model.predict(test_input_data)
    
    # if correct_normalization:
    #     test_predictions_denormalized = hf.denormalize_data(
    #         test_predictions, logaritmization=False,
    #         data_max=data_max[target_column_name[0]],
    #         data_min=data_min[target_column_name[0]])
    # else:
    #     test_predictions_denormalized = hf.denormalize_data(
    #         test_predictions, logaritmization=False,
    #         data_max=data_test_max[target_column_name[0]],
    #         data_min=data_test_min[target_column_name[0]])
    
    test_true_prices = true_prices[len(train_df)+len(val_df)+final_past_window_examples:-1]
    # capital, sharpe, cumulative_capital = PL.PL_backtesting_log_returns(
    #     test_true_prices.reshape(len(test_true_prices), 1),
    #     test_predictions_denormalized,
    #     threshold=treshold_in_PL)
    
    capital, sharpe, cumulative_capital = PL.PL_backtesting_log_returns(
        test_true_prices.reshape(len(test_true_prices), 1)/multiplication_constant,
        test_predictions/multiplication_constant,
        threshold=treshold_in_PL)
    
    R2 = r2_score(test_label_data/multiplication_constant, test_predictions/multiplication_constant)
    
    os.chdir(pics_source_directory)
    
    fig = plt.figure(constrained_layout=False)
    fig.suptitle(ticker)
    gs1 = fig.add_gridspec(nrows=5, ncols=1)
    
    ax1 = fig.add_subplot(gs1[:-2, :])
    # ax1.plot(test_label_data_denormalized, color='blue', label='skutočné log. výnosy')
    # ax1.plot(test_predictions_denormalized, color='black', label='predikcie')
    # ax1.axhspan(ymin=treshold_in_PL, ymax=np.max([np.max(test_predictions_denormalized), np.max(test_label_data_denormalized)]) , facecolor='#C6EFCE')
    # ax1.axhspan(ymin=np.min([np.min(test_predictions_denormalized), np.min(test_label_data_denormalized)]), ymax=-treshold_in_PL, facecolor='#FFC7CE')
    ax1.plot(test_label_data/multiplication_constant, color='blue', label='skutočné log. výnosy')
    ax1.plot(test_predictions/multiplication_constant, color='black', label='predikcie')
    ax1.axhspan(ymin=treshold_in_PL, ymax=np.max([np.max(test_predictions), np.max(test_label_data)])/multiplication_constant , facecolor='#C6EFCE')
    ax1.axhspan(ymin=np.min([np.min(test_predictions), np.min(test_label_data)])/multiplication_constant, ymax=-treshold_in_PL, facecolor='#FFC7CE')
    ax1.legend()
    
    ax2 = fig.add_subplot(gs1[3:5, :])
    ax2.plot(cumulative_capital, color='blue', label='vyvoj investovaneho kapitalu v %')
    ax2.legend()
    
    plt.savefig(ticker+'.png', dpi=1000)
    plt.show()
    
    
    evaluate_model = model.evaluate(test_input_data, test_label_data)
    # [test capital, test sharpe, test mae, test mse]
    results = [evaluate_model[0], evaluate_model[2], capital, sharpe, R2]
    return(results)    


columns_all_results = ['layer no', 'neurons in layer', 'past window', 'indicators_used',
        'val capital', 'val sharpe', 'val loss', 'val MSE']

columns_final_results = ['ticer', 'layer no', 'neurons in layer', 'val loss', 'val mse',
                         'val capital', 'val sharpe', 'indicators', 'past windows',
                         'test loss', 'test mse', 'test R2' ,'test PL', 'test sharpe']
row = 1
pd.DataFrame([] ,columns=columns_final_results).to_excel(writer_best_model, index=False, header=True)    

for tick in tickers:
    os.chdir(data_source_directory)
    df = pd.read_excel(tick.rstrip('\n')+'.xlsx')

    # df = df[['adj_close',
    #          'adj_high',
    #          'adj_low',
    #          'adj_open']]

    # df = df.rename(columns={'adj_close' : 'Close',
    #                         'adj_high' : 'High',
    #                         'adj_low' : 'Low',
    #                         'adj_open' : 'Open'})
    
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
    
    # df = df.dropna()

    log_returns_close = np.diff(np.log(df['Close']))
    log_returns_open = np.diff(np.log(df['Open']))
    log_returns_high = np.diff(np.log(df['High']))
    log_returns_low = np.diff(np.log(df['Low']))


    df_log_returns = pd.DataFrame(np.array([log_returns_close,
                                            log_returns_open,
                                            log_returns_high,
                                            log_returns_low]).T,
                                  columns=('Close', 'Open', 'High', 'Low'))
    df_length = len(df_log_returns)
    TRAIN_SPLIT = int(np.round(train_split_percentage * df_length))
    VAL_SPLIT = int(TRAIN_SPLIT + np.round(val_split_percentage * df_length))
    
    treshold_in_PL = np.quantile(np.abs(log_returns_close[:TRAIN_SPLIT]), q=0.2)
    true_price_check = np.array(df['Close'][1:])
    
    
    
    indicators = [[None], ['MA_4', 'MA_9', 'MA_18'], ['BBANDS_18'], ['MACD']]
    
    model_params, final_indicators, final_layer_number, final_neurons_in_layer, \
        final_val_loss, final_val_mean_squared_error, \
            final_past_window_examples, val_capital, val_sharpe = train_model(
                df_log_returns, true_prices=true_price_check, ticker=tick.rstrip('\n'),
                indicators=indicators,
                TRAIN_SPLIT=TRAIN_SPLIT, VAL_SPLIT=VAL_SPLIT,
                past_windows_min=10, past_windows_max=50, past_window_step=10,
                future_window_predictions=future_window_predictions,
                layer_no_min=0, layer_no_max=8,
                patience=2, BATCH=BATCH, max_epochs=30,
                random_drop_net_config_treshhold=None,
                target_column_name=target_column_name,
                final_model_destination=models_source_directory,
                treshold_in_PL=0, multiplication_constant=multiplication_constant)
    
    pd.DataFrame(model_params, columns=columns_all_results).to_excel(writer_all_results, sheet_name=tick.rstrip('\n'))
    
    # otestovanie finalneho modela
    final_results = test_final_model(
        df_log_returns, true_prices=true_price_check, ticker=tick.rstrip('\n'),
        indicators=final_indicators,
        final_past_window_examples=final_past_window_examples,
        TRAIN_SPLIT=TRAIN_SPLIT, VAL_SPLIT=VAL_SPLIT,
        BATCH=BATCH, final_model_destination=models_source_directory,
        treshold_in_PL=0, multiplication_constant=multiplication_constant)
    
    best_model_results = np.array([
        tick.rstrip('\n'), final_layer_number, final_neurons_in_layer,
        final_val_loss, final_val_mean_squared_error, val_capital, val_sharpe,
        final_indicators, final_past_window_examples, final_results[0],
        final_results[1], final_results[4] , final_results[2], final_results[3]], dtype='object_')
    
    pd.DataFrame(best_model_results.reshape(1, 14)).to_excel(writer_best_model, index=False, header=False, startrow=row)
    row += 1

os.chdir(models_source_directory)      
writer_all_results.save()
writer_best_model.save()
