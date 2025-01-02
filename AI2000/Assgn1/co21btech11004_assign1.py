


class node():
    '''
    Represents a node in the decision tree
    '''
    def __init__(self, feature=None, thr=None, left=None, right=None, perd_label=None):
        self.feature = feature
        self.thr = thr
        self.left = left
        self.right = right
        self.perd_label = perd_label


# In[6]:


class DecisionTree():
    '''
    Decision Tree Classifier
    '''
    def __init__(self, min_samples=2, max_depth=10):
        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def entropy(self, data):
        '''
        Calculate entropy of a dataset
        '''
        unique, counts = np.unique(data, return_counts=True)
        prob = counts / len(data)
        return -np.sum(prob * np.log2(prob))
    
    def info_gain(self, root, left, right):
        '''
        Calculate information gain
        '''
        wl = len(left) / len(root)
        wr = len(right) / len(root)
        return self.entropy(root) - wl * self.entropy(left) - wr * self.entropy(right)
    
    def temp_split(self, data, feature, thr):
        '''
        Split data into left and right based on threshold
        '''
        left = data[data[feature] <= thr]
        right = data[data[feature] > thr]
        return left, right
    
    def best_split(self, data):
        '''
        Find the best split
        '''
        best_split = {'ig': -1e9, 'feature': None, 'thr': None, 'left': None, 'right': None}
        for idx in range(len(data.columns) - 1):
            for thr in np.unique(data.iloc[:, idx]):
                left, right = self.temp_split(data, data.columns[idx], thr)
                if len(left) > 0 and len(right) > 0:
                    ig = self.info_gain(data.iloc[:, -1], left.iloc[:, -1], right.iloc[:, -1])
                    if ig > best_split['ig']:
                        best_split['ig'] = ig
                        best_split['feature'] = data.columns[idx]
                        best_split['thr'] = thr
                        best_split['left'] = left
                        best_split['right'] = right

        return best_split
    
    
    def set_label(self, data):
        '''
        Set the label of a node
        '''
        unique, counts = np.unique(data, return_counts=True)
        return unique[np.argmax(counts)]
    
    def build_tree(self, data, depth=0):
        '''
        Build the decision tree
        '''
        if len(data) >= self.min_samples and depth <= self.max_depth:
            split = self.best_split(data)
            if split['ig'] > 0:
                left = self.build_tree(split['left'], depth + 1)
                right = self.build_tree(split['right'], depth + 1)
                return node(feature=split['feature'], thr=split['thr'], left=left, right=right)
        
        return node(perd_label=self.set_label(data.iloc[:, -1]))
        
    def fit(self, X_train, y_train):
        '''
        Fit the model
        '''
        self.root = self.build_tree(pd.concat([X_train, y_train], axis=1))

    def predict(self, X_test):
        '''
        Predict the label
        '''
        y_pred = []
        for i in range(len(X_test)):
            node = self.root
            while node.perd_label is None:
                if X_test[node.feature].iloc[i] <= node.thr:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node.perd_label)
        return y_pred
    
    def accuracy(self, y_test, y_pred):
        '''
        Calculate accuracy
        '''
        return np.sum(y_test == y_pred) / len(y_test)   
    

# fit the model
# model = DecisionTree(1, 100)
model = DecisionTree(2, 5)
model.fit(X_train, y_train)

# predict the label
y_pred = model.predict(X_test)

# calculate accuracy
accuracy = model.accuracy(y_test, y_pred)
print(f'Accuracy: {accuracy}')

