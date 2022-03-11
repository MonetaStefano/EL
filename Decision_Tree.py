import numpy as np

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, w_value=None, t_value=None):
        #attributes of decision nodes
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.w_value = w_value
        #attribute of leaf nodes
        self.t_value = t_value
        
class Decision_tree:
    #class_or_reg attribute can either be 'calssification' or 'regression'
    #split_type attribute can either be 'gini' or 'entropy'
    def __init__(self, class_or_reg, split_type='gini', max_depth=2, min_sample_split=1):
        
        self.root = None
        self.max_depth = max_depth
        self.opt = class_or_reg
        self.split_type = split_type 
        self.min_sample_split = min_sample_split
        
    def gini_classification(self, y):
        
        labels = np.unique(y)
        gini = 0
        prob_sum = 0
        for l in labels:
            prob = len(y[y == l]) / len(y)
            prob_sum += prob**2
            gini = 1 - prob_sum
        return gini
    
    def entropy_classification(self, y):
        
        labels = np.unique(y)
        entropy = 0
        for l in labels:
            prob = len(y[y == l]) / len(y)
            if prob == 0: # when no element in one label, return 0
                return 0
            else:
                entropy += -prob * np.log2(prob)
        return entropy
    
    def se_regression(self, ytrue, yhat): #for regression use this to split, gini or entropy do not work 

        r = ytrue - yhat 
        return np.sum(r**2)
    
    def best_split(self, x, y):
        
        #create a dictionary to store best split
        best_split = {}
        data = np.c_[x, y]
        #set the initial gini large enough to make comparision
        i_value = 10000  
        
        for feature_index in range(x.shape[1]):
            
            feature_values = data[:, feature_index]
            possible_split = np.unique(feature_values)
            
            # loop over all the feature values present in the data
            for s in possible_split:
                
                # get current split
                data_left = data[data[:, feature_index] <= s]
                data_right = data[data[:, feature_index] > s]
                
                #classification using gini
                if (self.opt == 'classification' and self.split_type == 'gini'):
                    value_left = self.gini_classification(data_left[:, -1:])
                    value_right = self.gini_classification(data_right[:, -1:])
                    w_value = (len(data_left[:, -1:])*value_left + len(data_right[:-1:])*value_right)/len(data)
                    
                #classification using entropy
                elif (self.opt == 'classification' and self.split_type == 'entropy'):
                    value_left = self.entropy_classification(data_left[:, -1:])
                    value_right = self.entropy_classification(data_right[:, -1:])
                    w_value = (len(data_left[:, -1:])*value_left + len(data_right[:-1:])*value_right)/len(data)
                    
                #regression
                else:
                    value_left = self.se_regression(data_left[:, -1:], np.mean(data_left[:, -1:]))
                    value_right = self.se_regression(data_right[:, -1:], np.mean(data_right[:, -1:]))
                    w_value = value_left + value_right
              
                # update the best split
                if i_value > w_value:
                    best_split['feature_index'] = feature_index
                    best_split['threshold'] = s
                    best_split['data_left'] = data_left
                    best_split['data_right'] = data_right
                    best_split['w_value'] = w_value
                    i_value = w_value

        return best_split

    def build_tree(self, x, y, curr_depth=0):
        
        # split until stopping conditions are met
        if len(x) >= self.min_sample_split and curr_depth <= self.max_depth:
            
            # find the best split
            best_split = self.best_split(x, y)
            if best_split['w_value']>0:
                # recursion on the left tree
                left_subtree = self.build_tree(best_split['data_left'][:, :-1], best_split['data_left'][:, -1:], curr_depth+1)
                # recursion on the right tree 
                right_subtree = self.build_tree(best_split['data_right'][:, :-1], best_split['data_right'][:, -1:], curr_depth+1)
                # return decision node
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['w_value'])
            
        leaf_value = max(list(y), key=list(y).count)
        # return leaf node
        return Node(t_value=leaf_value)
    
    def fit(self, x, y):
        
        self.root = self.build_tree(x, y)
        
    def print_tree(self, tree=None, indent='  '):
        
        if not tree:
            tree = self.root
            
        if tree.t_value is not None:
            print(tree.t_value) 
        else:
            print('feature_'+str(tree.feature_index), '<=', round(tree.threshold, 5))
            print(indent, 'left: ', end='')
            self.print_tree(tree.left, indent + indent)
            print(indent, 'right: ', end='')
            self.print_tree(tree.right, indent + indent)
    
    def make_prediction(self, x, tree):
        
        if tree.t_value != None: 
            return tree.t_value
        
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
    
    def predict(self, x):
        
        preditions = [self.make_prediction(ele, self.root) for ele in x]
        return preditions