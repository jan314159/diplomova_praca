# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:53:17 2020

@author: janpi

vypocet PL, strategia vytvorena na zaklade neuronky, ak je predikovana cena vacsia,
tak kupujem, mensia predavam
"""


import numpy as np

# funkcia pre vypocet PL a sharpovho pomeru pre predikciu cien
def PL_backtesting_v2(true_future, predictions, threshold=0.03, 
                      initial_capital=100_000, risk_free=0):
    """
    

    Parameters
    ----------
    true_future : skutocne ceny, ake boli
    predictions : predikcia 
    initial_capital : pociatocny kapital na zaciatku. The default is 100_000.
    risk_free :  The default is 0.

    Returns
    -------
    PL pri danej predikcii a sharpeho pomer

    """
    # vypocitame, ci predikujeme oproti dnesku narast, alebo pokles, preto posunieme
    # vektor predikcii az od druhej, a v skutocnych sa nedivame na posledny
    change_next_day_prediction_vs_today_price = true_future[:-1] - predictions[1:]
    
    # indikatory buy a sell, ak je zmena kladna, teda predikuje sa vacsia hodnota, 
    # ako dnes, tak buy signal, ak zaporny, tak sell
    buy_indicator = (change_next_day_prediction_vs_today_price < -threshold*true_future[:-1])
    sell_indicator = (change_next_day_prediction_vs_today_price > threshold*true_future[:-1])
    
    # vytvorenie klasifikatora, narastu, poklesu, alebo ostania na tej istej urovni
    # ak buy ->>> +1, sell ->>> -1, no change ->>> 0
    buy_sell_indicator = np.zeros(change_next_day_prediction_vs_today_price.shape)
    buy_sell_indicator[buy_indicator] = 1
    buy_sell_indicator[sell_indicator] = -1
    
    # v prvy den obchodvania
    capital = initial_capital
    PL_day = 0
    position = 0
    RETURN = []
    day_return = 0
    cumulative_capital = []
    cumulative_capital.append(0)
    
    # for first day
    # pozicia, kolko longneme/shortujeme akcii
    position = (buy_sell_indicator[0] * capital) / true_future[0]
    
    # pre dni od prveho do predposledneho
    for i in range(1, len(true_future)-1):
        # vypocet zisku z predchadzajucej pozicie
        if(position>0):
            PL_day = position * true_future[i] - capital
        if(position<0):
            PL_day = position * true_future[i] + capital
        if(position==0):
            PL_day = 0
        
        day_return = (capital + PL_day) / capital - 1
        RETURN.append(day_return)
        # novy ziskany kapital
        capital = capital + PL_day
        # nova pozicia
        position = (buy_sell_indicator[i] * capital) / true_future[i]
        cumulative_capital.append((capital-initial_capital)*100 / initial_capital)

    # pre posledny den
    if(position>0):
        PL_day = position * true_future[-1] - capital
    if(position<0):
        PL_day = position * true_future[-1] + capital
    if(position==0):
        PL_day = 0
    capital = capital + PL_day
    cumulative_capital.append((capital-initial_capital)*100 / initial_capital)
    
    
    sharpe = np.sqrt(250) * (np.array(RETURN).mean(axis=0) - risk_free) / np.array(RETURN).std(axis=0)
    return(capital, sharpe, buy_sell_indicator, cumulative_capital)

# pridany spread
def PL_backtesting_v3(true_future, predictions, threshold=0.03,
                      initial_capital=100_000, risk_free=0, spread=0):
    """
    

    Parameters
    ----------
    true_future : skutocne ceny, ake boli
    predictions : predikcia 
    initial_capital : pociatocny kapital na zaciatku. The default is 100_000.
    risk_free :  The default is 0.

    Returns
    -------
    PL pri danej predikcii a sharpeho pomer

    """
    # vypocitame, ci predikujeme oproti dnesku narast, alebo pokles, preto posunieme
    # vektor predikcii az od druhej, a v skutocnych sa nedivame na posledny
    change_next_day_prediction_vs_today_price = true_future[:-1] - predictions[1:]
    
    # indikatory buy a sell, ak je zmena kladna, teda predikuje sa vacsia hodnota, 
    # ako dnes, tak buy signal, ak zaporny, tak sell
    buy_indicator = (change_next_day_prediction_vs_today_price < -threshold*true_future[:-1])
    sell_indicator = (change_next_day_prediction_vs_today_price > threshold*true_future[:-1])
    
    # vytvorenie klasifikatora, narastu, poklesu, alebo ostania na tej istej urovni
    # ak buy ->>> +1, sell ->>> -1, no change ->>> 0
    buy_sell_indicator = np.zeros(change_next_day_prediction_vs_today_price.shape)
    buy_sell_indicator[buy_indicator] = 1
    buy_sell_indicator[sell_indicator] = -1
    
    # v prvy den obchodvania
    capital = initial_capital
    PL_day = 0
    position = 0
    RETURN = []
    day_return = 0
    cumulative_capital = []
    cumulative_capital.append(0)
    
    # for first day
    # pozicia, kolko longneme/shortujeme akcii
    position = (buy_sell_indicator[0] * capital) / true_future[0]
    
    # pre dni od prveho do predposledneho
    for i in range(1, len(true_future)-1):
        # vypocet zisku z predchadzajucej pozicie
        if(position>0):
            PL_day = position * true_future[i] * (1-spread) - capital
        if(position<0):
            PL_day = position * true_future[i] * (1+spread) + capital
        if(position==0):
            PL_day = 0
        
        day_return = (capital + PL_day) / capital - 1
        RETURN.append(day_return)
        # novy ziskany kapital
        capital = capital + PL_day
        # nova pozicia
        position = (buy_sell_indicator[i] * capital) / true_future[i]
        cumulative_capital.append((capital-initial_capital)*100 / initial_capital)

    # pre posledny den
    if(position>0):
        PL_day = position * true_future[-1] * (1-spread) - capital
    if(position<0):
        PL_day = position * true_future[-1] * (1+spread) + capital
    if(position==0):
        PL_day = 0
    capital = capital + PL_day
    cumulative_capital.append((capital-initial_capital)*100 / initial_capital)
    
    
    sharpe = np.sqrt(250) * (np.array(RETURN).mean(axis=0) - risk_free) / np.array(RETURN).std(axis=0)
    return(capital, sharpe, buy_sell_indicator, cumulative_capital)


# funkcia pre vypocet PL a sharpovho pomeru pre predikciu vynosov
def PL_backtesting_log_returns(true_future, predictions,
                           threshold=0.005, initial_capital=100_000, risk_free=0):
    """
    

    Parameters
    ----------
    true_future : skutocne ceny, ake boli
    predictions : predikcia log vynosov
    initial_capital : pociatocny kapital na zaciatku. The default is 100_000.
    risk_free :  The default is 0.

    Returns
    -------
    PL pri danej predikcii a sharpeho pomer

    """

    # indikatory buy a sell, ak je zmena kladna, teda predikuje sa vacsia hodnota, 
    # ako dnes, tak buy signal, ak zaporny, tak sell
    buy_indicator = (predictions[1:] > threshold*np.log(true_future[:-1]))
    sell_indicator = (predictions[1:] < -threshold*np.log(true_future[:-1]))
    
    # vytvorenie klasifikatora, narastu, poklesu, alebo ostania na tej istej urovni
    # ak buy ->>> +1, sell ->>> -1, no change ->>> 0
    buy_sell_indicator = np.zeros(buy_indicator.shape)
    buy_sell_indicator[buy_indicator] = 1
    buy_sell_indicator[sell_indicator] = -1
    
    # v prvy den obchodvania
    capital = initial_capital
    PL_day = 0
    position = 0
    RETURN = []
    day_return = 0
    cumulative_capital = []
    cumulative_capital.append(0)
    
    # for first day
    # pozicia, kolko longneme/shortujeme akcii
    position = (buy_sell_indicator[0] * capital) / true_future[0]
    
    # pre dni od prveho do predposledneho
    for i in range(1, len(true_future)-1):
        # vypocet zisku z predchadzajucej pozicie
        if(position>0):
            PL_day = position * true_future[i] - capital
        if(position<0):
            PL_day = position * true_future[i] + capital
        if(position==0):
            PL_day = 0
        
        day_return = (capital + PL_day) / capital - 1
        RETURN.append(day_return)
        # novy ziskany kapital
        capital = capital + PL_day
        # nova pozicia
        position = (buy_sell_indicator[i] * capital) / true_future[i]
        cumulative_capital.append((capital-initial_capital)*100 / initial_capital)

    # pre posledny den
    if(position>0):
        PL_day = position * true_future[-1] - capital
    if(position<0):
        PL_day = position * true_future[-1] + capital
    if(position==0):
        PL_day = 0
    capital = capital + PL_day
    cumulative_capital.append((capital-initial_capital)*100 / initial_capital)
    
    
    sharpe = np.sqrt(250) * (np.array(RETURN).mean(axis=0) - risk_free) / np.array(RETURN).std(axis=0)
    return(capital, sharpe, cumulative_capital)

# pridany spread
def PL_backtesting_log_returns_v1(true_future, predictions,
                           threshold=0.005, initial_capital=100_000, risk_free=0, spread=0):
    """
    

    Parameters
    ----------
    true_future : skutocne ceny, ake boli
    predictions : predikcia log vynosov
    initial_capital : pociatocny kapital na zaciatku. The default is 100_000.
    risk_free :  The default is 0.

    Returns
    -------
    PL pri danej predikcii a sharpeho pomer

    """

    # indikatory buy a sell, ak je zmena kladna, teda predikuje sa vacsia hodnota, 
    # ako dnes, tak buy signal, ak zaporny, tak sell
    buy_indicator = (predictions[1:] > threshold*np.log(true_future[:-1]))
    sell_indicator = (predictions[1:] < -threshold*np.log(true_future[:-1]))
    
    # vytvorenie klasifikatora, narastu, poklesu, alebo ostania na tej istej urovni
    # ak buy ->>> +1, sell ->>> -1, no change ->>> 0
    buy_sell_indicator = np.zeros(buy_indicator.shape)
    buy_sell_indicator[buy_indicator] = 1
    buy_sell_indicator[sell_indicator] = -1
    
    # v prvy den obchodvania
    capital = initial_capital
    PL_day = 0
    position = 0
    RETURN = []
    day_return = 0
    cumulative_capital = []
    cumulative_capital.append(0)
    
    # for first day
    # pozicia, kolko longneme/shortujeme akcii
    position = (buy_sell_indicator[0] * capital) / true_future[0]
    
    # pre dni od prveho do predposledneho
    for i in range(1, len(true_future)-1):
        # vypocet zisku z predchadzajucej pozicie
        if(position>0):
            PL_day = position * true_future[i] * (1-spread) - capital
        if(position<0):
            PL_day = position * true_future[i] * (1+spread) + capital
        if(position==0):
            PL_day = 0
        
        day_return = (capital + PL_day) / capital - 1
        RETURN.append(day_return)
        # novy ziskany kapital
        capital = capital + PL_day
        # nova pozicia
        position = (buy_sell_indicator[i] * capital) / true_future[i]
        cumulative_capital.append((capital-initial_capital)*100 / initial_capital)

    # pre posledny den
    if(position>0):
        PL_day = position * true_future[-1] * (1-spread) - capital
    if(position<0):
        PL_day = position * true_future[-1] * (1+spread) + capital
    if(position==0):
        PL_day = 0
    capital = capital + PL_day
    cumulative_capital.append((capital-initial_capital)*100 / initial_capital)
    
    
    sharpe = np.sqrt(250) * (np.array(RETURN).mean(axis=0) - risk_free) / np.array(RETURN).std(axis=0)
    return(capital, sharpe, cumulative_capital)
