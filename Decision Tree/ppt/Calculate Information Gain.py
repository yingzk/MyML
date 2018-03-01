def calc_gain(dataset, attr, target_attr):
    """
    Calculate Information Gain
    @param dataset: Dataset
    @param attr: super attribute
    @param target_attr: tarrget attribute
    """
    
    #super information entropy and sub-class entropy
    mclass_entropy = calc_entroy(dataset, attr)
    sclass_entropy = 0.0
    
    for i in dataset[target_attr].unique():
        # different attribute's different class's sub set
        sub_dataset = dataset[dataset[target_attr] == i]
        sclass_entropy += len(sub_dataset) / len(dataset) * calc_entroy(sub_dataset, attr)
    
    return mclass_entropy - sclass_entropy
