def build_tree(attr_rem, data_rem, X, Y, attr_dict, trees, fa_label):  
	"""
	@param attr_rem: remain attribute
	@param data_rem: remain dataset
	@param X: recording
	@param Y: Good?
	@param attr: attribute dictionary
	@param trees: records of the built tree
	@param fa_label: father node label
	"""
	
    divided_by = None  
    sons = []  
    label = None  
    condition = None  
      
    count = cal_count(data_rem, Y) # number of every kind  
    total_cnt = len(data_rem) # total number of remain data  
    max_kind = max(count, key=lambda x: count[x]) # kind which has the largest number  
    ent = cal_ent(count, total_cnt)  
  
    if len(data_rem) == 0:  
        label = fa_label  
    elif attr_rem is None or len(count) == 1: # remain attribute is empty or all data is one same kind, then this is a leaf node  
        label = max_kind  
    else:  
        best_divided_res = []  
        best_condition = []  
        best_divided_by = None  
        best_info_gain = -100  
  
        for attr in attr_rem:  
            is_float = type(X[data_rem[0]][attr]) == float  
            if is_float:  
                points = sorted(attr_dict[attr])  
                divided_points = [(points[i]+points[i+1])/2 for i in range(len(points)-1)]  
                for divided_point in divided_points:  
                    divided_res = []  
                    divided_res.append([i for i in data_rem if X[i][attr] <= divided_point])  
                    divided_res.append([i for i in data_rem if X[i][attr] > divided_point])  
                    info_gain = cal_info_gain(ent, divided_res, total_cnt, Y)  
                    if info_gain > best_info_gain:  
                        best_info_gain = info_gain  
                        best_divided_res = divided_res  
                        best_divided_by = attr  
                        best_condition = divided_point  
            else:  
                divided_res = []  
                for name in attr_dict[attr]:  
                    divided_res.append([j for j in data_rem if X[j][attr]==name])  
                info_gain = cal_info_gain(ent, divided_res, total_cnt, Y)  
                if info_gain > best_info_gain:  
                        best_info_gain = info_gain  
                        best_divided_res = divided_res  
                        best_divided_by = attr  
                        best_condition = attr_dict[attr]  
          
        divided_by = best_divided_by  
        is_float = type(X[data_rem[0]][divided_by]) == float  
        if not is_float:  
            attr_rem = [i for i in attr_rem if i != divided_by] # can't use remove! behaviour not expected when dealing with utf-8  
  
        sons = [build_tree(attr_rem, i, X, Y, attr_dict, trees, max_kind) for i in best_divided_res]  
        condition = best_condition  
  
    trees.append(Node(divided_by, condition, sons, label))  
    return len(trees)-1  
