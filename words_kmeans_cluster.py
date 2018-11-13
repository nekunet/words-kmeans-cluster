import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import scipy.spatial.distance
import scipy.cluster.hierarchy
import matplotlib
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA


model = gensim.models.KeyedVectors.load_word2vec_format('ここにバイナリ化した学習データのパスを入力', binary=True)

wordvec = [] # 単語のベクトルのリスト
vocab_new = [] # 辞書にある単語のみを格納するリスト

# for test
vocab = ["斉藤", "リンゴ", "シマウマ", "東京", "ライオン", "名古屋", "ミカン", "ウシ", "メロン", "田中", "横浜", "鈴木"]

for x in vocab:
    try:
        wordvec.append(model.get_vector(x))
    except KeyError:
        print(x, "を無視します")
        continue

    vocab_new.append(x)

kmeans_model = KMeans(n_clusters=4, verbose=1, max_iter=30)
kmeans_model.fit(wordvec)

cluster_labels = kmeans_model.labels_
cluster_to_words = defaultdict(list)
for cluster_id, word in zip(cluster_labels, vocab_new):
    cluster_to_words[cluster_id].append(word)

for words in cluster_to_words.values():
    print(words)

font = {"family": "Spica Neue P"}  # Windows用
plt.rc('font', **font)

# PCAで次元削減
pca = PCA(n_components=2)
wordvec_r = pca.fit_transform(wordvec)

fig, ax = plt.subplots()

# 結果を散布図にプロット
for (i, label) in enumerate(kmeans_model.labels_):
    if label == 0:
        ax.scatter(wordvec_r[i, 0], wordvec_r[i, 1], c='red')
        ax.annotate(vocab_new[i], xy=(wordvec_r[i, 0], wordvec_r[i, 1]), size=8)
    elif label == 1:
        ax.scatter(wordvec_r[i, 0], wordvec_r[i, 1], c='blue')
        ax.annotate(vocab_new[i], xy=(wordvec_r[i, 0], wordvec_r[i, 1]), size=8)
    elif label == 2:
        ax.scatter(wordvec_r[i, 0], wordvec_r[i, 1], c='green')
        ax.annotate(vocab_new[i], xy=(wordvec_r[i, 0], wordvec_r[i, 1]), size=8)
    elif label == 3:
        ax.scatter(wordvec_r[i, 0], wordvec_r[i, 1], c='orange')
        ax.annotate(vocab_new[i], xy=(wordvec_r[i, 0], wordvec_r[i, 1]), size=8)
plt.show()
