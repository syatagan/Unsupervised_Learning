import pandas as pd
import seaborn as sns
import datetime as dt
from Src.utils import check_df
from Src.utils import cat_summary
from Src.utils import num_summary
#from Src.utils import grab_col_names

def grab_col_names(xdf,xcat_th = 10 , xcar_th = 20):
    cat_cols = [col for col in xdf.columns if xdf[col].dtypes in ["bool", "object", "category"]]
    num_but_cat = [col for col in xdf.columns if xdf[col].dtypes in ["int64", "float64"]
                   and xdf[col].nunique() < xcat_th]
    cat_but_car = [col for col in xdf.columns if xdf[col].dtypes in ["object", "category"]
                   and xdf[col].nunique() > xcar_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in xdf.columns if str(xdf[col].dtypes) in ["float64","int64"]
                and col not in cat_cols]

    print(f"Observations : {xdf.shape[0]}")
    print(f"Variables : {xdf.shape[1]}")
    print(f"Cat_cols : {len(cat_cols)}")
    print(f"Num_cols : {len(num_cols)}")
    print(f"Cat_But_car : {len(cat_but_car)}")
    print(f"Num_but-cat : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


### OPTIONS
pd.set_option("display.width",300)
pd.set_option("display.max_columns",None)

#############################################################
"""Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz."""

df_ = pd.read_csv("Datasets/flo_data_20k.csv")
df = df_.copy()
df.head()
#############################################################
"""
Adım 2: Veri setinde
a. İlk 10 gözlem,
b. Değişken isimleri,
c. Betimsel istatistik,
d. Boş değer,
e. Değişken tipleri, incelemesi yapınız."""
############################################################

check_df(df,xrow_count=10)
cat_cols , num_cols, cat_but_car = grab_col_names(df)

print("KATEGORIK DEĞİŞKENLER" )
print(cat_cols)
print("NUMERIC DEĞİŞKENLER" )
print(num_cols)
print("KATEGORIK ama KARDINAL DEĞİŞKENLER")
print(cat_but_car)
print("KATEGORIK DEĞİŞKEN İNCELEMESİ")
for col in cat_cols:
    cat_summary(df, col)
for col in num_cols:
    num_summary(df, col)

#############################################################
"""
Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. 
Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
"""

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_amount"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.loc[:, ["master_id","total_order_num","total_amount"] ].head()
df.head()
#############################################################
"""
Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz."""

df.dtypes
date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.dtypes
df.head()
#############################################################
"""
Adım 5: Alışveriş kanallarındaki müşteri sayısının, 
toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
"""
df["order_channel"].value_counts()
df["last_order_channel"].value_counts()
df.loc[df["last_order_channel"] == "Offline","order_channel"].value_counts()
df.loc[df["customer_value_total_ever_offline"] > 0 ,["order_channel","order_num_total_ever_offline","customer_value_total_ever_online"]]

df.groupby("order_channel")["total_amount","total_order_num"].agg("sum")
#############################################################
"""Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız."""

df[["master_id","total_amount"]].sort_values("total_amount",axis=0,ascending=False).head(10)
#############################################################
"""Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız."""

df[["master_id","total_order_num"]].sort_values("total_order_num",axis=0,ascending=False).head(10)
#############################################################
"""
Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

## ???
"""
#############################################################
"""
Görev 2: RFM Metriklerinin Hesaplanması
Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz
"""

df["last_order_date"].max()
to_date = dt.datetime(2021,6,2)

df["Recency"] = df["last_order_date"].apply( lambda x : (to_date - x).days)
df["Frequency"] = df["total_order_num"]
df["Monetary"] = df["total_amount"]

rfm =df[["master_id","Recency","Frequency","Monetary"]]
rfm.columns = ["master_id","recency","frequency","monetary"]
rfm.head()
rfm.dtypes
rfm.describe().T

##################################################################
"""
Görev 3: RF Skorunun Hesaplanması
Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz. 
Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
"""

rfm["recency_score"] = pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"],5,labels=[1,2,3,4,5])
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

###################################################################
"""
Görev 4: RF Skorunun Segment Olarak Tanımlanması
Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz
"""
# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
################################################################
"""
Görev 5: Aksiyon Zamanı
Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.
"""
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm[["recency","frequency","monetary","segment"]].groupby("segment").agg("mean")

"""
a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. 
Dahil ettiği markanın ürün fiyatları genel müşteri
tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve 
ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
iletişime geçmek isteniliyor. 
Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
"""
rfm.head()
df.head()

target_segment_list= ["champions","loyal_customers"]
df_kadin = df.loc[(df.interested_in_categories_12.str.contains("KADIN") ),"master_id"]

df_target_customers = rfm.loc[rfm.segment.isin(target_segment_list) &
                                rfm.master_id.isin(df_kadin.values),"master_id"]
df_target_customers.to_csv("1.csv")

"""
b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, 
uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor. 
Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz
"""

target_segment_list= ["new_customers","about_to_sleep","cant_loose"]
df_erkek_cocuk = df[(df["interested_in_categories_12"].str.contains("ERKEK")) |
                        (df["interested_in_categories_12"].str.contains("COCUK"))]["master_id"]

df_target_customers = rfm.loc[rfm.segment.isin(target_segment_list) &
                              rfm.master_id.isin(df_erkek_cocuk.values),"master_id"]
df_target_customers.to_csv("2.csv")


### merge yöntemi ile
rfm2 = rfm[(rfm["segment"].isin(target_segment_list)) ]
df2 = df_erkek_cocuk
df_v2 = pd.merge(df2, rfm2, on="master_id",how="inner")
df_v2