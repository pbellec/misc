import numpy as np


def simu_maps(n_voxel, n_replication, n_subject, within, between):
    """Simulate replication of brain maps within and between subjects
    parameters:
        n_voxel: the number of voxel value in a map
        n_replication: the number of map replication per subject
        n_subject: the number of subjects
        within: the strength of the within-subject effect
        between: the strength of the between-subject effect

    Returns: y (the maps), gt (ground truth partition into subjects)
    """
    noise = np.random.normal(
        size=[n_replication, n_voxel]
    )  # Some Gaussian random noise
    gt = np.zeros(shape=[n_replication, 1])  # Ground truth clusters
    y = np.zeros(noise.shape)  # The final time series
    ind = np.linspace(
        0, n_replication, n_subject + 1, dtype="int"
    )  # The indices for each cluster
    map_group = np.random.normal(
        size=[1, n_voxel]
    )  # a group map common to all subjects
    for ss in range(0, n_subject):  # loop over subjects
        cluster = range(
            ind[ss], ind[ss + 1]
        )  # replications for that particular subject
        map_subject = np.random.normal(size=[1, n_voxel])  # subject-specific map
        y[cluster, :] = (
            noise[cluster, :]
            + between * np.repeat(map_group, ind[ss + 1] - ind[ss], 0)
            + within * np.repeat(map_subject, ind[ss + 1] - ind[ss], 0)
        )  # y = noise + a * signal
        gt[cluster] = ss  # Adding the label for cluster in ground truth
    return y, gt


def part2adj(part):
    """Convert a clustering vector, with integer labels, into an adjancency matrix"""
    part = np.reshape(part, [part.shape[0], 1])
    adj = np.repeat(part, part.shape[0], 1) == np.repeat(
        part.transpose(), part.shape[0], 0
    )
    return adj.astype(int)


def map2corr(y, gt):
    corr_matrix = np.corrcoef(y)
    subjects_effect = part2adj(gt)
    iu = np.mask_indices(corr_matrix.shape[0], np.triu, k=1)
    mask_sub = subjects_effect[iu]
    data = np.arctanh(corr_matrix[iu])
    return data, mask_sub > 0


def eff_size(data, mask_sub):
    n1 = np.sum(mask_sub > 0)
    n2 = np.sum(mask_sub == 0)
    mean1 = np.mean(data[mask_sub > 0])
    mean2 = np.mean(data[mask_sub == 0])
    std1 = np.std(data[mask_sub > 0])
    std2 = np.std(data[mask_sub == 0])
    std_pool = np.sqrt(((n1 - 1) * std1 + (n2 - 1) * std2) / (n1 + n2))
    d = (mean1 - mean2) / std_pool
    return d


def compute_diff(data, mask_sub):
    avg_within = np.mean(data[mask_sub>0])
    avg_between = np.mean(data[mask_sub==0])
    return avg_within - avg_between


def permutation_test(data, mask_sub, n_permutation=10000):
    diff_permutation = []
    diff_data = compute_diff(data, mask_sub)
    for per in np.arange(n_permutation):
        mask_permutation = np.random.permutation(mask_sub)
        diff_permutation.append(compute_diff(data, mask_permutation))
    p_permutation = (np.sum((diff_permutation >= diff_data).astype('float')) + 1) / (n_permutation + 1)
    return p_permutation, diff_permutation, diff_data


def simu_stats(
    n_simulation, n_permutation, n_voxel, n_replication, n_subject, within, between
):
    p_values = []
    cohen_d = []
    for ss in range(n_simulation):
        y, gt = simu_maps(
            n_voxel=100, n_replication=40, n_subject=4, within=within, between=0.5
        )
        data, mask_sub = map2corr(y, gt)
        d = eff_size(data, mask_sub)
        p, diff_permutation, diff = permutation_test(data, mask_sub, n_permutation=n_permutation)
        p_values.append(p)
        cohen_d.append(d)
    return np.array(p_values), cohen_d
