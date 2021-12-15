import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


def decision_tree(train, test, use_data_list):
    # 「train」の目的変数と説明変数の値を取得
    target = train['Survived'].values
    features_one = train[use_data_list].values

    # 決定木の作成
    my_tree_one = tree.DecisionTreeClassifier()
    my_tree_one = my_tree_one.fit(features_one, target)

    # 「test」の説明変数の値を取得
    test_features = test[use_data_list].values

    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    return my_tree_one.predict(test_features)


def random_forest(train, test, use_data_list, random_state=4096):
    target = train['Survived'].values
    features_one = train[use_data_list].values

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(features_one, target)

    # 「test」の説明変数の値を取得
    test_features = test[use_data_list].values

    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    return clf.predict(test_features)


def make_kaggle_answer(my_prediction, test):
    # PassengerIdを取得
    PassengerId = np.array(test['PassengerId']).astype(int)
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(my_prediction, PassengerId, columns=['Survived'])
    # my_tree_one.csvとして書き出し
    my_solution.to_csv('my_tree_one.csv', index_label=['PassengerId'])
