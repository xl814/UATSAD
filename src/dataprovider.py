
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

def load_SWAT_FIT502(abspath = False):
    if abspath == False:
        test_df = pd.read_csv('datasets/SWaT/SWaT_test_original.csv')

    test_np = test_df['FIT502'].to_numpy() # AIT501
    test_label = test_df['label'].to_numpy()
    test = test_np[0: 4000].reshape(-1, 1)
    training = test_np[4000: 16000].reshape(-1, 1)
    valid = test_np[16000: 18000].reshape(-1, 1)


    # training = (training - np.mean(training)) / (np.std(training))
    # test = (test - np.mean(test)) / np.std(test)
    # valid = (valid - np.mean(valid)) / np.std(valid)

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

def load_SWAT_PIT502(abspath = False):
    if abspath == False:
        test_df = pd.read_csv('datasets/SWaT/SWaT_test_original.csv')
    else:
        test_df = pd.read_csv('/home/new_lab/test/ensemble_bae/datasets/SWaT/SWaT_test_original.csv')
    
    test_np = test_df['PIT502'].to_numpy() 
    test_label = test_df['label'].to_numpy()

    test = test_np[0: 4000].reshape(-1, 1)
    training = test_np[4000: 16000].reshape(-1, 1)
    valid = test_np[16000: 18000].reshape(-1, 1)


    # training = (training - np.mean(training)) / (np.std(training))
    # test = (test - np.mean(test)) / np.std(test)
    # valid = (valid - np.mean(valid)) / np.std(valid)

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")
    
    return training, test, valid

def load_SMAP_P1(abspath = False):
    if abspath == False:
        test_data = np.load('datasets/SMAP/tmp/P-1_test.npy')
        train_data = np.load('datasets/SMAP/tmp/P-1_train.npy')
    else:
        test_data = np.load('/home/new_lab/test/ensemble_bae/datasets/SMAP/tmp/P-1_test.npy')
        train_data = np.load('/home/new_lab/test/ensemble_bae/datasets/SMAP/tmp/P-1_train.npy')


    training = train_data[:, 0].reshape(-1, 1)
    # training = test_data[5000:, 0].reshape(-1, 1)
    test = test_data[2000: 5000, 0].reshape(-1, 1)
    valid = test_data[: 2000, 0].reshape(-1, 1)


    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)
    # print(f"raw shape: {readings.shape}")
    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")
    return training, test, valid

def load_SMAP_E1():
    raw_data = np.load('datasets/SMAP/E-1.npy')

    readings = raw_data[0: 6500, 0]
    idx_split = [0, 3500, 4500]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    test = readings[idx_split[2]: ].reshape(-1,1)

    # labels (500, 530, ) (1100, 1500)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)
    print(f"raw shape: {raw_data.shape}")
    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")
    # print(f"lables shape: {lables.shape}")
    return training, test, valid



def load_SMAP_E13():
    raw_data = np.load('datasets/SMAP/E-13.npy')

    readings = raw_data[0: 7000, 0]
    idx_split = [0, 4000, 5000]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    test = readings[idx_split[2]: ].reshape(-1,1)

    # labels (300, 400) (600, 650) (1450, 1550)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)
    print(f"raw shape: {raw_data.shape}")
    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")
    # print(f"lables shape: {lables.shape}")
    return training, test, valid

def load_NAB_Ambient():

    def load_data():
        ambient_temp = pd.read_csv('ambient_temperature_system_failure.csv')
        anomalies_label = ['2013-12-22 20:00:00', '2014-04-13 09:00:00']
        
        anomalies_idx = []
        for label in anomalies_label:
            anomalies_idx.append(ambient_temp[ambient_temp['timestamp'] == label].index[0])

        return ambient_temp['value'].values, anomalies_idx

    readings, idx_anomaly = load_data()
    print(readings.shape, idx_anomaly)
    # split reading into training and test sets
    idx_split = [0, 3200, 6400]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    test = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    valid = readings[idx_split[2]: ].reshape(-1,1)

    # standardize the dataf
    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    idx_anomaly_test = np.asarray(idx_anomaly) - idx_split[1]

    print(f"training set: {training.shape}")
    print(f"valid set: {valid.shape}")
    print(f"test set: {test.shape}")
    print(f"anomalies (in test set): {idx_anomaly_test}")
    return training, test, valid

def load_NAB_Texi(abspath=False):
    if abspath == False:
        data_dir = 'datasets/NAB-known-anomaly/csv-files/nyc_taxi.csv'
    else:
        data_dir = np.load('/home/new_lab/test/ensemble_bae/datasets/NAB-known-anomaly/csv-files/nyc_taxi.csv')

    def load_data(data_dir):
        raw = pd.read_csv(data_dir)
        anomalies_label = [         
            "2014-11-01 19:00:00",
            "2014-11-27 15:30:00",
            # "2014-12-25 15:00:00",  # NOTE: we ignored SOME due to train/valid/test split strategy
            # "2015-01-01 01:00:00",
            # "2015-01-27 00:00:00"
            ]
        
        anomalies_idx = []
        for label in anomalies_label:
            anomalies_idx.append(raw[raw['timestamp'] == label].index[0])

        return raw['value'].values, anomalies_idx

    readings, idx_anomaly = load_data(data_dir)
    print(readings.shape, idx_anomaly)
    # split reading into training and test sets
    idx_split = [0, 4500, 5500, 7500]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    test = readings[idx_split[2]: idx_split[3]].reshape(-1,1)

    # standardize the data
    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    idx_anomaly_test = np.asarray(idx_anomaly) - idx_split[2]

    print(f"training set: {training.shape}")
    print(f"valid set: {valid.shape}")
    print(f"test set: {test.shape}")
    print(f"anomalies (in test set): {idx_anomaly_test}")

    return training, test, valid

def load_NAB_Texi2(abspath=False):
    '''
        The difference between this and the above is that we use the whole dataset for visualization
    '''
    if abspath == False:
        data_dir = 'datasets/NAB-known-anomaly/csv-files/nyc_taxi.csv'
    else:
        data_dir = '/home/new_lab/test/ensemble_bae/datasets/NAB-known-anomaly/csv-files/nyc_taxi.csv'

    def load_data(data_dir):
        raw = pd.read_csv(data_dir)
        anomalies_label = [         
            "2014-11-01 19:00:00",
            "2014-11-27 15:30:00",
            "2014-12-25 15:00:00",  # NOTE: we ignored SOME due to train/valid/test split strategy
            "2015-01-01 01:00:00",
            "2015-01-27 00:00:00"
            ]
        
        anomalies_idx = []
        for label in anomalies_label:
            anomalies_idx.append(raw[raw['timestamp'] == label].index[0])

        return raw['value'].values, anomalies_idx

    readings, idx_anomaly = load_data(data_dir)
    print(readings.shape, idx_anomaly)
    # split reading into training and test sets
    idx_split = [0, 4500, 5500, 7500]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    test = readings[idx_split[2]: ].reshape(-1,1)

    # standardize the data
    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    idx_anomaly_test = np.asarray(idx_anomaly) - idx_split[2]

    print(f"training set: {training.shape}")
    print(f"valid set: {valid.shape}")
    print(f"test set: {test.shape}")
    print(f"anomalies (in test set): {idx_anomaly_test}")

    return training, test, valid

def load_NAB_Machine():
    data_dir = 'datasets/NAB-known-anomaly/csv-files/machine_temperature_system_failure.csv'
    def load_data(data_dir):
        raw = pd.read_csv(data_dir)

        anomalies_label = [         
            "2013-12-11 06:00:00",
            "2013-12-16 17:25:00",
            #  "2014-01-28 13:55:00", 
            #  "2014-02-08 14:30:00"
            ]
        
        anomalies_idx = []
        for label in anomalies_label:
            anomalies_idx.append(raw[raw['timestamp'] == label].index[0])

        return raw['value'].values, anomalies_idx

    readings, idx_anomaly = load_data(data_dir)
    print(readings.shape, idx_anomaly)
    # split reading into training and test sets
    idx_split = [0, 2000, 4500, 10000]
    training = readings[idx_split[2]: idx_split[3]].reshape(-1,1)
    valid = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    test = readings[idx_split[1]: idx_split[2]].reshape(-1,1)

    # standardize the data
    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    idx_anomaly_test = np.asarray(idx_anomaly) - idx_split[1]

    print(f"training set: {training.shape}")
    print(f"valid set: {valid.shape}")
    print(f"test set: {test.shape}")
    print(f"anomalies (in test set): {idx_anomaly_test}")

    return training, test, valid


def load_SMD_Machine2_1():

    raw_test = np.genfromtxt('datasets/SMD/test/machine-2-1.txt',
                            dtype=np.float64,
                            delimiter=',')

    raw_train = np.genfromtxt('datasets/SMD/train/machine-1-1.txt',
                            dtype=np.float64,
                            delimiter=',')


    raw_labels = np.genfromtxt('datasets/SMD/labels/machine-2-1.txt',
                            dtype=np.float64,
                            delimiter=',')

    # [[15849, 16368],[16963, 17517], [18071, 18528], [19367, 20088], [20786,21195]]
    # [[6506, 6530], [7900, 7960], [9340, 9380]]
    # idx_split = [0, 12000, 15000, 19000]

    idx_split = [0, 5000, 6000, 9600]
    readings = raw_test[:, 31]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    test = readings[idx_split[2]: idx_split[3]].reshape(-1,1)

    labels = raw_labels.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    valid = scaler.transform(valid)
    test = scaler.transform(test)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")
    print(f"lables shape: {labels.shape}")
    return training, test, valid

def load_SMD_Machine1_3():
    raw_test = np.genfromtxt('datasets/SMD/test/machine-1-3.txt',
                         dtype=np.float64,
                         delimiter=',')

    print(raw_test.shape)
    idx_split = [1280, 4500, 6000, 7000]
    readings = raw_test[:, 11]
    training = readings[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = readings[idx_split[2]: idx_split[3]].reshape(-1,1)
    test = readings[idx_split[1]: idx_split[2]].reshape(-1,1)

    # valid = training[(int)(0.8 * training.shape[0]): ].reshape(-1, 1)
    # training = training[: (int)(0.8 * training.shape[0])].reshape(-1, 1)
    # test = raw_test[15000: 18000, 0].reshape(-1,1)


    # labels (500, 530, ) (1100, 1500)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    valid = scaler.transform(valid)
    test = scaler.transform(test)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")

    return training, test, valid



def load_SMD_Machine3_4():
    raw_test = np.genfromtxt('datasets/SMD/test/machine-3-4.txt',
                         dtype=np.float64,
                         delimiter=',')

    print(raw_test.shape)
    idx_split = [2500, 5000, 6000, 12000]
    readings = raw_test[:, 1]
    training = readings[idx_split[2]: idx_split[3]].reshape(-1,1)
    valid = readings[idx_split[1]: idx_split[2]].reshape(-1,1)
    test = readings[idx_split[0]: idx_split[1]].reshape(-1,1)

    # valid = training[(int)(0.8 * training.shape[0]): ].reshape(-1, 1)
    # training = training[: (int)(0.8 * training.shape[0])].reshape(-1, 1)
    # test = raw_test[15000: 18000, 0].reshape(-1,1)


    # labels (500, 530, ) (1100, 1500)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    valid = scaler.transform(valid)
    test = scaler.transform(test)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")

    return training, test, valid


def load_MSL_F7():
    raw_data_train = np.load('datasets/MSL/F-7_train.npy')[:, 0]
    raw_data_test = np.load('datasets/MSL/F-7_test.npy')[:, 0]

    training = raw_data_train.reshape(-1,1)
    valid = raw_data_test[4000: ].reshape(-1,1)
    test = raw_data_test[: 4000].reshape(-1,1)

    # labels (500, 530, ) (1100, 1500)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")

    return training, test, valid

def load_MSL_P11():
    raw_data_train = np.load('datasets/MSL/P-11_train.npy')[:, 0]
    raw_data_test = np.load('datasets/MSL/P-11_test.npy')[:, 0]

    training = raw_data_train.reshape(-1,1)
    valid = raw_data_test[2500: ].reshape(-1,1)
    test = raw_data_test[:2500].reshape(-1,1)

    # labels (500, 530, ) (1100, 1500)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")

    return training, test, valid

def load_UCR_InternalBleeding16():

    raw_data_test = np.genfromtxt('datasets/UCR/135_UCR_Anomaly_InternalBleeding16_1200_4187_4199.txt',
								dtype=np.float64,
								delimiter=',')
    print(raw_data_test.shape)
    idx_split = [0, 4000, 6000, 7500] # [0, 3500, 4500, 5500]
    training = raw_data_test[idx_split[0]: idx_split[1]].reshape(-1,1)
    valid = raw_data_test[idx_split[2]: idx_split[3]].reshape(-1,1)
    test = raw_data_test[idx_split[1]: idx_split[2]].reshape(-1,1)

    # labels (4185, 4200)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")

    return training, test, valid

def load_UCR_InternalBleeding17():
    raw_data_test = np.genfromtxt('datasets/UCR/136_UCR_Anomaly_InternalBleeding17_1600_3198_3309.txt',
                                    dtype=np.float64,
                                    delimiter=',')
    print(raw_data_test.shape)
    data_label = np.load('datasets/UCR/135_labels.npy')
    idx_split = [0, 1000, 4000] # [1000, 2000, 4000]
    training = raw_data_test[idx_split[2]: ].reshape(-1,1)
    valid = raw_data_test[idx_split[0]: idx_split[1]].reshape(-1,1)
    test = raw_data_test[idx_split[1]: idx_split[2]].reshape(-1,1)

    # labels (3200, 3300)
    # lables = np.ones(idx_split[1] - idx_split[0]) # 2150 - 2350 is anomaly
    # lables[2150:2350] = 0

    scaler = StandardScaler()
    scaler.fit(training)
    training = scaler.transform(training)
    test = scaler.transform(test)
    valid = scaler.transform(valid)

    print(f"training shape: {training.shape}")
    print(f"test shape: {test.shape}")
    print(f"valid shape: {valid.shape}")

    return training, test, valid



