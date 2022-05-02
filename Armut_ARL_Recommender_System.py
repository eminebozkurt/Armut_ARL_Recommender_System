# Görev 1: Veriyi Hazırlama

# Adım 1: armut_data.csv dosyasını okutunuz.

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
import datetime as dt

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_csv("week5/hw1/armut_data.csv")
df = df_.copy()
df.head()


# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df["New_Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

df.head()


# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. \
# Örneğin; 25446 id'li müşteri 2017'in 8.ayında aldığı 4_5, 48_5, 6_7, 47_7 hizmetler bir sepeti;
# 2017'in 9.ayında aldığı 17_5, 14_7 hizmetler başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek
# ID adında yeni bir değişkene atayınız. Elde edilmesi gereken çıktı:

# df["New_Date"] = [col[:7] for col in df["CreateDate"].astype(str)]

df["CreateDate"] = pd.to_datetime(df["CreateDate"])

df["New_CreateDate_YM"] = df['CreateDate'].dt.year.astype(str) + "_" + df['CreateDate'].dt.month.astype(str)
df.head()

df["New_SepetID"] = df["UserId"].astype(str) + "_" + df["New_CreateDate_YM"]

df.head()

#Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
# Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.

df.describe().T
df.shape
df.isnull().sum()

df.head()

df.pivot_table(columns=["New_Hizmet"], index=["New_SepetID"], values=["ServiceId"], aggfunc="count").head(3)

invoice_product_df = df.pivot_table(columns=["New_Hizmet"],
                                    index=["New_SepetID"],
                                    values=["ServiceId"],
                                    aggfunc="count").fillna(0).applymap(lambda x: 1 if x > 0 else 0)

invoice_product_df = df.groupby(['New_SepetID', 'New_Hizmet']). \
    agg({"New_Hizmet": "count"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

invoice_product_df = df.groupby(['New_SepetID', 'New_Hizmet'])['New_Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df

invoice_product_df.columns = invoice_product_df.columns.droplevel(0)

invoice_product_df.head()

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(invoice_product_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.shape

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules.head()

#rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.01) & (rules["lift"] > 1)]. \
#sort_values("confidence", ascending=False)


sorted_rules = rules.sort_values("lift", ascending=False)

sorted_rules.head()

# Adım 3: arl_recommender fonksiyonunu kullanarak son 1 ay içerisinde 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

for idx, product in enumerate(sorted_rules["antecedents"].head()):
  print(idx, "_", product)
  print(list(product))

def arl_recommender(rules_df, product_id, rec_count=1):
    recommendation_list = []
    sorted_rules = rules.sort_values("lift", ascending=False)
    for idx, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
              recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 5)

df.head()

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["CreateDate"].max()# Timestamp('2011-12-09 12:50:00')

df["days"] = (df["CreateDate"].max() - df["CreateDate"]).dt.days
df.head()
df.info()

df = df[df["days"] <= 30]

df_new = df.groupby(['SepetID', 'Hizmet']). \
    agg({"Hizmet": "count"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(df_new,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"] > 0.01) & (rules["confidence"] > 0.01) & (rules["lift"] > 1)]. \
sort_values("confidence", ascending=False)

product_id = "2_0"
recommendation_list = []
sorted_rules = rules.sort_values("lift", ascending=False)

for idx, product in enumerate(sorted_rules["antecedents"]): # antecendent tuple olduğu için listeye çevirelim ve liste içinde arayalım:
    for j in list(product):
        if j == product_id: # bu yakaladığımız integer değerin indexi ne ise (idx) consequentte onu arayacağız, bulduğumuz satırlar için ilk ürünü [0] önerelim
            recommendation_list.append(list(sorted_rules.iloc[idx]["consequents"])[0])
            recommendation_list = list(dict.fromkeys(recommendation_list))

recommendation_list[0:3]


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

arl_recommender(rules, "2_0", 1)