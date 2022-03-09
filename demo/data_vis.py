import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from tsnecuda import TSNE
from MulticoreTSNE import MulticoreTSNE
# from bdd_kitti_extract import kitti_bdd_map


def auroc(y, score):
    fpr, tpr, thresholds = roc_curve(y, score)
    print(auc(fpr, tpr))
    return fpr, tpr, thresholds


if __name__ == '__main__':
    fc = np.load('fcs.npy')
    logit = np.load('logits.npy')
    soft = np.load('softmax.npy')
    pred = np.load('preds.npy')
    flag = np.load('flags.npy')

    fc_bdd = np.load('city_fcs.npy')
    logit_bdd = np.load('city_logits.npy')
    soft_bdd = np.load('city_softmax.npy')
    pred_bdd = np.load('city_preds.npy')
    flag_bdd = np.load('city_flags.npy')

    auroc(flag, soft)
    auroc(flag_bdd, soft_bdd)

    # reduce = PCA(n_components=2)
    # reduce = TSNE(n_components=2)
    reduce = MulticoreTSNE(n_components=2, n_jobs=-1)
    fc_ld = reduce.fit_transform(fc_bdd)

    corr = np.where(flag_bdd == 1)[0]
    corr_pred = pred_bdd[corr]
    fc_ldcorr = fc_ld[corr]

    fig, ax = plt.subplots(figsize=(16, 12))

    for i in range(8):
        ax.scatter(fc_ldcorr[corr_pred == i, 0], fc_ldcorr[corr_pred == i, 1], s=4, alpha=0.3, label='{}'.format(i))
    # ax.scatter(fc_ld[flag_bdd == 0, 0], fc_ld[flag_bdd == 0, 1], alpha=0.5, c='gray')
    plt.legend()
    plt.savefig('/home/kengo/Pictures/bdd.pdf')
    plt.show()
