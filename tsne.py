from sklearn.manifold import TSNE
#from sklearn.datasets import load_iris,load_digits
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np

# digits = load_digits()
# print(digits.data.shape)
# print(digits.target)
# print(digits.data)
#X_tsne = TSNE(n_components=2,random_state=33).fit_transform(digits.data)
#X_pca = PCA(n_components=2).fit_transform(digits.data)

def read_19lou_vec():
    data=[]
    with open('19lou_vec_out.txt') as f:
        f.readline()
        while True:
            line=f.readline().strip().split(' ')
            if len(line)>1:
                tid=line[0]
                vec=[float(x) for x in line[1:]]
                #print(tid)
                #print(vec)
                data.append(vec)
            else:
                break
    #print(data)
    return np.array(data)

data=read_19lou_vec()
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
#X_pca = PCA(n_components=2).fit_transform(data)
ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

plt.figure(figsize=(10, 10))
plt.subplot(111)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],label="t-SNE")#, c=digits.target
plt.legend()
# plt.subplot(122)
# plt.scatter(X_pca[:, 0], X_pca[:, 1],label="PCA") #, c=digits.target
# plt.legend()
plt.savefig('images/digits_tsne-pca.png', dpi=120)
plt.show()