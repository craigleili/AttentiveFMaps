import os
import os.path as osp
import sys
import numpy as np
import scipy.linalg
from tqdm import tqdm

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils.misc import KNNSearch

try:
    import pynndescent
    index = pynndescent.NNDescent(np.random.random((100, 3)), n_jobs=2)
    del index
    ANN = True
except ImportError:
    ANN = False

# https://github.com/RobinMagnet/pyFM


def FM_to_p2p(FM, eigvects1, eigvects2, use_ANN=False):
    if use_ANN and not ANN:
        raise ValueError('Please install pydescent to achieve Approximate Nearest Neighbor')

    k2, k1 = FM.shape

    assert k1 <= eigvects1.shape[1], \
        f'At least {k1} should be provided, here only {eigvects1.shape[1]} are given'
    assert k2 <= eigvects2.shape[1], \
        f'At least {k2} should be provided, here only {eigvects2.shape[1]} are given'

    if use_ANN:
        index = pynndescent.NNDescent(eigvects1[:, :k1] @ FM.T, n_jobs=8)
        matches, _ = index.query(eigvects2[:, :k2], k=1)
        matches = matches.flatten()
    else:
        tree = KNNSearch(eigvects1[:, :k1] @ FM.T)
        matches = tree.query(eigvects2[:, :k2], k=1).flatten()

    return matches


def p2p_to_FM(p2p, eigvects1, eigvects2, A2=None):
    if A2 is not None:
        if A2.shape[0] != eigvects2.shape[0]:
            raise ValueError("Can't compute pseudo inverse with subsampled eigenvectors")

        if len(A2.shape) == 1:
            return eigvects2.T @ (A2[:, None] * eigvects1[p2p, :])

        return eigvects2.T @ A2 @ eigvects1[p2p, :]

    return scipy.linalg.lstsq(eigvects2, eigvects1[p2p, :])[0]


def zoomout_iteration(eigvects1, eigvects2, FM, step=1, A2=None, use_ANN=False):
    k2, k1 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p = FM_to_p2p(FM, eigvects1, eigvects2, use_ANN=use_ANN)
    FM_zo = p2p_to_FM(p2p, eigvects1[:, :new_k1], eigvects2[:, :new_k2], A2=A2)

    return FM_zo


def zoomout_refine(eigvects1,
                   eigvects2,
                   FM,
                   nit=10,
                   step=1,
                   A2=None,
                   subsample=None,
                   use_ANN=False,
                   return_p2p=False,
                   verbose=False):
    k2_0, k1_0 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit*step1 <= eigvects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {eigvects1.shape[1]} are provided"
    assert k2_0 + nit*step2 <= eigvects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {eigvects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_zo = FM.copy()

    ANN_adventage = False
    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:
        ANN_adventage = use_ANN and (FM_zo.shape[0] > 90) and (FM_zo.shape[1] > 90)

        if use_subsample:
            FM_zo = zoomout_iteration(eigvects1[sub1], eigvects2[sub2], FM_zo, A2=None, step=step, use_ANN=ANN_adventage)

        else:
            FM_zo = zoomout_iteration(eigvects1, eigvects2, FM_zo, A2=A2, step=step, use_ANN=ANN_adventage)

    if return_p2p:
        p2p_zo = FM_to_p2p(FM_zo, eigvects1, eigvects2, use_ANN=False)
        return FM_zo, p2p_zo

    return FM_zo
