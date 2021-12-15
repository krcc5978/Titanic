import DecisionTree.decision_tree as dt
import DecisionTree.load_data as ld
import Graph.Graph as g


train_data_path = '../titanic/train.csv'
test_data_path = '../titanic/test.csv'


def main():
    train_data, test_data = ld.load_data(train_data_path, test_data_path)

    g.hoge(train_data)

    # 読み込み時のデータテーブルの確認
    ld.check_data_tabel(train_data)
    ld.check_data_tabel(test_data)

    # 欠損値の代入
    ld.insert_missging_data(train_data)
    ld.insert_missging_data(test_data)
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

    # 欠損地などを埋めた後のデータテーブルの確認
    ld.check_data_tabel(train_data)
    ld.check_data_tabel(test_data)

    result = dt.random_forest(train_data, test_data, ['Pclass', 'Name', 'Sex', 'Fare', 'Embarked'])
    print(result)

    dt.make_kaggle_answer(result, test_data)


if __name__ == '__main__':
    main()
