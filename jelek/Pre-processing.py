import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#csv egymasba tevese meg label elott
def append_datas(df1):
    df1.to_csv("my_data.csv", mode='a', index = False)

#positive label letrehozas
def add_label_pos(data):
    length = data['RAW_TP9'].size

    new_column = []
    for i in range(length):
        new_column.append("Positive")

    return new_column


#negativ label letrahozas
def add_label_neg(data):
    length = data['RAW_TP9'].size

    new_column = []
    for i in range(length):
        new_column.append("Negativ")

    return new_column

#elso adathalmaz letrehozasa, ehhez adok majd hozza
def setup_data():

    df1 = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/mintak/Positive2.csv')

    creating_data_pos(df1)
    df = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/programok/jelek/temporal.csv')
    append_datas(df)

def append_new_data(df):
    df.to_csv('my_data.csv', index=False, mode='a', header=False)

#kirajzolas, meg nem mukodik
def print_data(data1):
    # kirajzolom a nyers adatot
    plt.figure(1)
    plt.plot(data1)
    plt.legend()
    plt.show()

    #kirajzolom a PSD(power spectrum density) utani eredmenyt
    #f, Pxx = sc.welch(data1)
    # plt.figure(2)
    # plt.plot(Pxx)
    # #plt.semilogy(Pxx)
    # #plt.imshow(Pxx, aspect= 'auto')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # wavelet transzformacio, meg nem sikeres
    # plt.figure(3)
    # cwtmatr, freqs = pywt.cwt(data1, 208, 'morl')
    #
    # plt.imshow(cwtmatr, aspect='auto')
    # plt.title("Spektrum morlet transzformációval")
    # plt.show()

#pozitiv label hozzaadas
def creating_data_pos(df2):

    data_ = df2.loc[:, 'RAW_TP9': 'RAW_TP10']
    data2 = data_.dropna()

    new_column = add_label_pos(data2)
    data2['label'] = new_column
    data2.to_csv("temporal.csv", index=False)

#neutral label letrehozas
def add_label_neut(data):
    length = data['RAW_TP9'].size

    new_column = []
    for i in range(length):
        new_column.append("Neutral")

    return new_column

#adathalmazhoz a neutral hozzaragasztasa
def creating_data_neut(df):

    data = df.loc[:, 'RAW_TP9': 'RAW_TP10']
    data2 = data.dropna()
    new_column = add_label_neut(data2)
    data2['label'] = new_column
    data2.to_csv("temporal.csv", index=False)

#a mar kesz csv-hez hozzadni az ujabb felcimkezett adatokat
def merge_data():
    merged =  pd.read_csv('C:/Users/Asus/Desktop/allamviszga/programok/jelek/temporal.csv')
    mer = merged.loc[:,'RAW_TP9': 'label']
    append_new_data(mer)

def creating_data_neg(df):
    data = df.loc[:, 'RAW_TP9': 'RAW_TP10']
    data2 = data.dropna()
    new_column = add_label_neg(data2)
    data2['label'] = new_column
    data2.to_csv("temporal.csv", index=False)

if __name__=='__main__':
    #setup_data()
    # df = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/programok/jelek/data_new.csv')
    # creating_data_neut(df)
    # merge_data()
    # df2 = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/mintak/Positive1.csv')
    # creating_data_pos(df2)
    # merge_data()
    # df3 = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/mintak/Negative1.csv')
    # creating_data_neg(df3)
    # merge_data()
    # df4 = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/mintak/Negative2.csv')
    # creating_data_neg(df4)
    # merge_data()
    # df5 = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/mintak/Neutral2.csv')
    # creating_data_neut(df5)
    # merge_data()
    # df5 = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/mintak/negative4.csv')
    # creating_data_neg(df5)
    # merge_data()
    df = pd.read_csv('C:/Users/Asus/Desktop/allamviszga/programok/jelek/my_data.csv')

    # megnezem , hogy hany darab adatom van mindegyikbol
    print(df['label'].value_counts())
