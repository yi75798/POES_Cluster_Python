#!/usr/bin/python
# -*- encoding: utf-8 -*-
# File    :   cluster.py
# Time    :   2021/12/15 14:31:17
# Author  :   Hsu, Liang-Yi
# Email:   yi75798@gmail.com
# Description : 2021輿情分析實習課資料分群程式，以TF-IDF做K-means分群

import re
from collections import Counter
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))

# 文本斷詞處理================================================================
# 原始文本
df = pd.read_csv('TextData.csv', encoding='UTF-8')  # Windows改成ANSI或ascii編碼試試

# 停用詞、使用者字典匯入
stopwords = [word.strip() for word in open('stopwords.txt', 'r')]
jieba.load_userdict('userdict.txt')

# 斷詞處理
df['words'] = None
for i in df.index:
    # 清除數字、換行符號、空白
    text = re.sub(r'[0-9%\n\s]+', '', df['Text'].loc[i])
    result = [seg for seg in jieba.cut(
        text, cut_all=False) if seg not in stopwords]

    df['words'].loc[i] = " ".join(result)

# 計算tf-idf================================================================
# 分詞器
vectorizer = CountVectorizer()
text = list(df['words'])  # 將所有斷詞結果合併

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(text))
word = vectorizer.get_feature_names()  # 所有單詞
weight = tfidf.toarray()  # tf-idf權值array

dtm = pd.DataFrame(weight, columns=word).set_index(df.index)  # DTM矩陣

# 將tf-idf向量存入df
df['tf-idf'] = None
for i in df.index:
    tf_idf = dtm.loc[i].values.tolist()
    df['tf-idf'].loc[i] = tf_idf

# K-means分群================================================================


def best_cluster(data, k_range=(2, 10)) -> str:
    '''
    :data: array: dtm array
    :k_range: tuple: 分群範圍(最小k, 最大k). defalt: (2, 10)
    :return: SSE & Silhouette score method plot. best selected K.
    '''
    k_range = range(k_range[0], k_range[1])
    distortions = []
    scores = []
    for i in k_range:
        kmeans = KMeans(n_clusters=i).fit(data)
        distortions.append(kmeans.inertia_)  # 誤差平方和 (SSE)
        scores.append(silhouette_score(data, kmeans.predict(data)))  # 側影係數

    selected_K = scores.index(max(scores)) + 2

    # 繪製誤差平方和圖 (手肘法)
    plt.subplot(121)
    plt.title('SSE (elbow method)')
    plt.plot(k_range, distortions)
    plt.plot(selected_K, distortions[selected_K - 2], 'go')  # 最佳解

    # 繪製係數圖
    plt.subplot(122)
    plt.title('Silhouette score')
    plt.plot(k_range, scores)
    plt.plot(selected_K, scores[selected_K - 2], 'go')  # 最佳解

    plt.tight_layout()
    plt.show()
    print(selected_K)

    return selected_K


k = int(best_cluster(weight, (2, 12)))

clf = KMeans(n_clusters=k).fit(weight)

df['cluster'] = list(clf.labels_)

# 輸出檔案
df.to_excel('文本分群結果.xlsx', encoding='UTF-8', index=False)
