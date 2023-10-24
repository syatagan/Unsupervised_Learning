import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import datetime as dt
from Src.utils import *
from Src.conf import *
plt.matplotlib.use('Qt5Agg')
def read_FLO_data():
    df = pd.read_csv("datasets/flo_data_20k.csv")
    check_df(df)
    return df
def prepare_data(df):
    cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)
    missing_value_analyser(df,df.columns)
    outlier_analyser(df, num_cols)
    # type conversions
    date_cols = [col for col in df.columns if 'date' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # creating new features
    # ## df["last_order_date"].max()
    analysis_date = dt.datetime(2021, 6, 2) # analysis date
#    df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
#   df["total_amount"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
    df["Recency"] = df["last_order_date"].apply(lambda x: (analysis_date - x).days)
    df["Tenure"] = df["first_order_date"].apply(lambda x: (analysis_date - x).days)
    # interested list split and create new features
    interest_list = df["interested_in_categories_12"].unique()
    interest_set = set()
    for xcat in interest_list:
        xcat = xcat.replace(" ","").replace("[","").replace("]","")
        split_list = xcat.split(sep=",")
        for xstr in split_list:
            if xstr != "" : interest_set.add(xstr)
    for xcategory in interest_set:
        df[xcategory] = df.apply(lambda x: 1 if xcategory in x["interested_in_categories_12"] else 0, axis=1)

    ## check for features
    cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)
    Correlation_Analysis(df, num_cols, plot = False)

    ## standardization
    sc = MinMaxScaler((0, 1))
    df[num_cols] = sc.fit_transform(df[num_cols])
    df = df.drop(["master_id","first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline","interested_in_categories_12"], axis=1)
    ohe_cols = ["order_channel","last_order_channel"]
    df = one_hot_encoder(df, ohe_cols)
    df.columns= df.columns.str.replace(" ","")
    return df
def Create_Model_Kmeans(df):
    # optimum küme sayısının bulunması
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    elbow.fit(df)
    elbow.show(block=True)
    xn_cluster = elbow.elbow_value_ ## optimum cluster sayısı

    kmeans = KMeans(n_clusters=xn_cluster, random_state=17).fit(df)
    df["kmeans_cluster_no"] = kmeans.labels_
    df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
    return df
def Create_Model_Hierarcihal(df):
    hc_average = linkage(df, "average")
    plt.figure(figsize=(10, 5))
    plt.title("Hiyerarşik Kümeleme Dendogramı")
    plt.xlabel("Gözlem Birimleri")
    plt.ylabel("Uzaklıklar")
    dendrogram(hc_average,
               leaf_font_size=10)
    plt.show(block=True)

    plt.figure(figsize=(7, 5))
    plt.title("Hiyerarşik Kümeleme Dendogramı")
    plt.xlabel("Gözlem Birimleri")
    plt.ylabel("Uzaklıklar")
    dendrogram(hc_average,
               truncate_mode="lastp",
               p=5,
               show_contracted=True,
               leaf_font_size=10)
    plt.show(block=True)

 # küme sayısını belirlemek
    plt.figure(figsize=(7, 5))
    plt.title("Dendrograms")
    dend = dendrogram(hc_average)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.axhline(y=0.6, color='b', linestyle='--')
    plt.show(block=True)
    ################################
    # Final Modeli Oluşturmak
    ################################
    cluster = AgglomerativeClustering(n_clusters=10, linkage="average")
    clusters = cluster.fit_predict(df)
    df["hi_cluster_no"] = clusters
    df["hi_cluster_no"] = df["hi_cluster_no"] + 1
    return df

def FLO_pipeline():
    set_display_configuration()
    df = read_FLO_data()
    df_tmp = prepare_data(df)
    df_tmp2 = Create_Model_Kmeans(df_tmp)
    df_final = Create_Model_Hierarcihal(df_tmp2)
    df_final = df
    df_final["hi_cluster_no"].value_counts()
    df_final["kmeans_cluster_no"].value_counts()

    df_final.loc[df["customer_segment"] == 0]["hi_cluster_no"].value_counts()
    df_final.groupby("hi_cluster_no").agg(["count","mean","median"])
    df_final.groupby("kmeans_cluster_no").agg(["count", "mean", "median"])
