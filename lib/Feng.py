import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.proportion import proportions_ztest


def outlier_thresholds(dataframe, column, q1=0.05, q3=0.95):
    """
    Aykiri degerlerin alt limitini ve ust limitini donduren fonksiyon
    Parameters
    ----------
    dataframe: Pandas.series
        Aykiri degerin bulunmasi istediginiz dataframe'i giriniz
    column: str
        Hangi degisken oldugunu belirtiniz
    q1:  float
        Alt limit ceyrekligini belirtin
    q3: float
        Ust limit ceyrekligini belirtin

    Returns
    -------
    low_th: float
        alt eşik değer
    up_th: float
        üst eşik değer
    """
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_th = quartile3 + 1.5 * iqr
    low_th = quartile1 - 1.5 * iqr
    return low_th, up_th


def check_outlier(dataframe, column):
    """
    Veri setindeki değişkenlerin içerisinde outlier değer var mı yok mu kontrol eder geriye bool değer döndürür.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    column: str
        Hangi değişken içerisinde sorgulama yapmak istiyorsak o değişkeni giriniz

    Returns
    -------


    """
    if pd.api.types.is_numeric_dtype(dataframe[column]):
        low_limit, up_limit = outlier_thresholds(dataframe, column)
        outliers = dataframe[(dataframe[column] > up_limit) | (dataframe[column] < low_limit)]
        return not outliers.empty
    else:
        raise ValueError(f"The column {column} is not numeric and cannot be checked for outliers.")


def grab_outliers(dataframe, col_name, index=False):
    """
    Bu fonksiyon bize ilgili değişkendeki aykırı değerleri döndürür.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col_name: str
        Hangi değişkenin bilgilerini istiyorsanız o değişkeni yazınız
    index: bool
        Fonksiyonun geriye aykırı değerleri dönmesini istiyorsanız True verin


    Returns
    -------

    """
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_specific_outlier(dataframe, col_name):
    """
    Bu fonskiyon istediğmiz kolondaki aykırı değerleri siler
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col_name: str
        dataframe deki aykırı değerleri silmek istediğimiz değişkenin ismi

    Returns
    -------
        df_without_outliers: dataframe
            Aykırı değerlerin silindiği dataframe i döner

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, col):
    """
    Belli bir değişkendeki aykırı değerleri baskılamak yerini alt ve üst limitlerle doldurmak için kullanınız.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col: str
        Baskılamak istediğiniz değişkenin ismi

    Returns
    -------
        dataframe: dataframe
            Baskılanmış yerini alt ve üst limitlerle doldurulmuş bir dataframe döndürür

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col)
    dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
    dataframe.loc[(dataframe[col] > up_limit), col] = up_limit


def lof(dataframe, neighbors=20, plot=False):
    """
    İstenilen veri setine LOF (Local Outlier Factor) yöntemini uygular.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    th_index: int
        İçerisine index bilgisi alır o indeksten sonraki verileri kırpar eğer bilgi girilmezse istenilen veriler getirilmez
    neighbors: int
        Komşuluk sayısını belirtiniz.

    Returns
    -------
        df_scores: np.Array
            Geriye değişkenlerin skorlarını döndürür.
    """
    dataframe = dataframe.select_dtypes(include=['float64', 'int64'])
    clf = LocalOutlierFactor(n_neighbors=neighbors)
    clf.fit_predict(dataframe)
    df_scores = clf.negative_outlier_factor_
    if plot:
        scores = pd.DataFrame(np.sort(df_scores))
        scores.plot(stacked=True, xlim=[0, 50], style='.-')
        plt.show()

    return df_scores


def lof_indexes(df, threshold):
    df_scores = lof(df)
    th = np.sort(df_scores)[threshold]
    return df[df_scores < th].index


def label_encoder(dataframe, binary_col, info=False):
    labelencoder = LabelEncoder()

    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    if info:
        d1, d2 = labelencoder.inverse_transform([0, 1])
        print(f'{binary_col}\n0:{d1}, 1:{d2}')
    return dataframe


def encode_all_binary_columns(dataframe, binary_cols, info=False):
    for col in binary_cols:
        label_encoder(dataframe, col, info)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def hypothesis_testing(df, new_feature, target):
    test_stat, pvalue = proportions_ztest(count=[df.loc[df[new_feature] == 1, target].sum(),
                                                 df.loc[df[new_feature] == 0, target].sum()],

                                          nobs=[df.loc[df[new_feature] == 1, target].shape[0],
                                                df.loc[df[new_feature] == 0, target].shape[0]])

    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
