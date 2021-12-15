import matplotlib.pyplot as plt


def correlation_coefficient(data):
    df_corr = data.corr()
    print(df_corr)


def hoge(data):

    pclass = data.groupby(['Pclass']).count()['Survived']
    pclass.plot(kind='bar')
    # a = data.groupby(['Pclass']).mean()["Survived"]
    # a.plot(kind='bar')
    # plt.bar(a)
    plt.show()
