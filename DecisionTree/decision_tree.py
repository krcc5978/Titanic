import pandas as pd
import numpy as np
from sklearn import tree

train_data_path = "../titanic/train.csv"
test_data_path = "../titanic/test.csv"


def missing_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


def prepare_data(table):
    table["Age"] = table["Age"].fillna(table["Age"].median())
    table["Embarked"] = table["Embarked"].fillna("S")

    table['Sex'] = table['Sex'].replace('male', 0)
    table['Sex'] = table['Sex'].replace('female', 1)
    table['Embarked'] = table['Embarked'].replace('S', 0)
    table['Embarked'] = table['Embarked'].replace('C', 1)
    table['Embarked'] = table['Embarked'].replace('Q', 2)


def decision_tree(train, test):
    # 「train」の目的変数と説明変数の値を取得
    target = train["Survived"].values
    features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

    # 決定木の作成
    my_tree_one = tree.DecisionTreeClassifier()
    my_tree_one = my_tree_one.fit(features_one, target)

    # 「test」の説明変数の値を取得
    test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    return my_tree_one.predict(test_features)


def check_survivor():
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    prepare_data(train)
    prepare_data(test)
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    print(missing_table(train))
    print(missing_table(test))

    result = decision_tree(train, test)

    print(result)
