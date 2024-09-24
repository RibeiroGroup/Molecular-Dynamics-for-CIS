
def EM_mode_exhaust(
        possible_kval_list,
        max_kval = None,
        ):

    """
    Exhautively generate all combination for mode vector
    """

    all_combs = combinations_with_replacement(
            possible_kval_list, 2
            )
    #generate all combinations of sorted integers, e.g. (0,1) or (1,1)

    modes_list = []

    for comb in all_combs:
        """
        Permuting element in each combination
        """
        comb = list(comb)
        comb_modes_list = []

        if np.sum(comb) < 1: 
            # skip (0,0) 
            continue

        comb_val = np.sum(np.array(comb)**2)
        if max_kval and comb_val > max_kval**2:
            # skipp combination that are above certain threshold
            continue

        comb_modes_list.append(np.array(comb))
        if comb[0] == comb[1]:
            pass
        else:
            comb_modes_list.append(
                    np.array([comb[1], comb[0]])
                    )
        #perm = set(permutations(comb))

        comb_modes_list = np.array(comb_modes_list)

        modes_list.append(comb_modes_list)

    modes_list = np.vstack(modes_list)
    modes_list = np.hstack([
        modes_list, np.zeros((len(modes_list),1))])

    return modes_list


def EM_mode_generate_(max_n, n_vec_per_kz = 1, min_n = 1):
    modes_list = []
    for i in range(min_n, max_n + 1):
        sample_range = np.arange(1,i + 1)
        if n_vec_per_kz < len(sample_range):
            ky = np.random.choice(
                    sample_range, size = n_vec_per_kz - 1, 
                    replace = False)
            ky = np.hstack([[0], ky])
        else: ky = sample_range

        kz = np.array([i] * len(ky)).reshape(-1,1)
        ky = ky.reshape(-1,1)
        kx = np.zeros(ky.shape)

        mode_vector = np.hstack([kx,ky,kz])
        modes_list.append(mode_vector)
    return np.vstack(modes_list)

def EM_mode_generate3(max_n,min_n = 1, max_n111 = None):
    mode_list = []
    for i in range(min_n, max_n+1):
        mode_list.append([i,0,0])
        mode_list.append([0,i,0])
        mode_list.append([0,0,i])

        """
        if max_n111 and i < max_n111:
            for j in np.arange(max(0,i-2),min(max_n,i+2)):
                mode_list.append([j,j,i])
                mode_list.append([j,i,j])
                mode_list.append([i,j,j])
        """

    return np.vstack(mode_list)

