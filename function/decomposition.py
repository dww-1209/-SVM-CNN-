from sklearn.decomposition import PCA

def pca(train_feature,test_feature):
    pca = PCA(n_components=100,random_state=0)
    train_reduced = pca.fit_transform(train_feature)
    test_reduced = pca.transform(test_feature)
    return train_reduced,test_reduced   