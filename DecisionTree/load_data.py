import numpy as np
import pandas as pd


def load_data(train_data_path, test_data_path, full_row_data=False, full_col_data=False):
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    if full_row_data:
        pd.set_option('display.max_rows', None)

    if full_col_data:
        pd.set_option('display.max_columns', None)

    return train, test


def check_data_tabel(data_table):
    print('------------ table head ------------')
    print(data_table.head())

    print('------------ table describe ------------')
    print(data_table.describe())

    print('------------ table correlation　coefficient ------------')
    df_corr = data_table.corr()
    print(df_corr)

    print('------------ table missing data ------------')
    print(missging_data(data_table))

    print('------------------------------------')


def missging_data(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    missing_data_table = pd.concat([null_val, percent], axis=1)
    missing_data_table_ren_columns = missing_data_table.rename(
        columns={0: '欠損数', 1: '%'})

    return missing_data_table_ren_columns


def insert_missging_data(table):
    for i, data in enumerate(table['Age']):
        if np.isnan(data):
            table['Age'][i] = np.random.normal(table['Age'].mean(), table['Age'].std())
    table['Embarked'] = table['Embarked'].fillna(np.random.randint(0, 3))

    table['Sex'] = table['Sex'].replace('male', 0)
    table['Sex'] = table['Sex'].replace('female', 1)

    table['Embarked'] = table['Embarked'].replace('S', 0)
    table['Embarked'] = table['Embarked'].replace('C', 1)
    table['Embarked'] = table['Embarked'].replace('Q', 2)

    table['Name'] = table['Name'].replace('(.*)Mr\.(.*)', 0, regex=True)
    table['Name'] = table['Name'].replace('(.*)Mrs\.(.*)', 1, regex=True)
    table['Name'] = table['Name'].replace('(.*)Miss\.(.*)', 2, regex=True)
    table['Name'] = table['Name'].replace('(.*)Master\.(.*)', 3, regex=True)
    table['Name'] = table['Name'].replace('(.*)Dr\.(.*)', 4, regex=True)
    table['Name'] = table['Name'].replace('(.*)Col\.(.*)', 5, regex=True)
    table['Name'] = table['Name'].replace('(.*)Mlle\.(.*)', 6, regex=True)
    table['Name'] = table['Name'].replace('(.*)Rev\.(.*)', 7, regex=True)
    table['Name'] = table['Name'].replace('(.*)Sir\.(.*)', 8, regex=True)
    table['Name'] = table['Name'].replace('(.*)Lady\.(.*)', 9, regex=True)
    table['Name'] = table['Name'].replace('(.*)Major\.(.*)', 10, regex=True)
    table['Name'] = table['Name'].replace('(.*)Mme\.(.*)', 11, regex=True)
    table['Name'] = table['Name'].replace('(.*)Don\.(.*)', 12, regex=True)
    table['Name'] = table['Name'].replace('(.*)Jonkheer\.(.*)', 13, regex=True)
    table['Name'] = table['Name'].replace('(.*)Countess\.(.*)', 14, regex=True)
    table['Name'] = table['Name'].replace('(.*)Capt\.(.*)', 15, regex=True)
    table['Name'] = table['Name'].replace('(.*)Ms\.(.*)', 16, regex=True)
    table['Name'] = table['Name'].replace('(.*)Dona\.(.*)', 17, regex=True)
