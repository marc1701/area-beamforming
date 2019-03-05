import eigenscape
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def multifold_classify(data, indices, labels, fold_info, n_folds=4):

    conf_size = len(labels)
    confmats = []
    confmat_sum = np.zeros((conf_size, conf_size))

    for i in range(n_folds):

        # split data and labels
        X = data[:,:-1]
        y = data[:,-1]

        scaler = StandardScaler()

        train_info = eigenscape.extract_info(fold_info +
            'fold' + str(i+1) + '_train.txt')
        test_info = eigenscape.extract_info(fold_info +
            'fold' + str(i+1) + '_test.txt')

        train_indices = eigenscape.vectorise_indices(train_info, indices)

        X_train = X[train_indices]
        y_train = y[train_indices]

        classifier = eigenscape.MultiGMMClassifier()
        classifier.fit(scaler.fit_transform(X_train), y_train)

        y_test, y_score = eigenscape.BOF_audio_classify(
            classifier, scaler.transform(X), y, test_info, indices)

        fold_confmat = eigenscape.plot_confusion_matrix(
            y_test, y_score, labels, plot=False)

        confmats.append(fold_confmat)
        confmat_sum += fold_confmat

    return confmats, confmat_sum
