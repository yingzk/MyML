def calc_entroy(dataset, target_attr):
    """
   Calculate information entropy
    @param dataset: dataset
    @param target_attr: target attribute
    """
    data_entropy = 0.0;
    for i in dataset[target_attr].unique():
        # count numbers of attribute
        # target attribute of i
        i_nums = len(dataset[dataset[target_attr] == i])
        data_entropy -= (i_nums / len(dataset)) * np.log2(i_nums / len(dataset))
    return data_entropy
