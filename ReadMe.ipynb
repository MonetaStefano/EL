{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f8a028f2",
   "metadata": {},
   "source": [
    "# ReadMe"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10e427a7",
   "metadata": {},
   "source": [
    "We build the decision tree from scratch based on the algorithm Classification and Regression Tree(CART). All the functions are wrapped in one class: Decision_tree. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "98995676",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Node():\n",
    "    def __init__(self, feature_index=None, threshold=None, left=None, right=None, w_value=None, t_value=None):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "831d6ace",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Decision_tree:\n",
    "    #class_or_reg attribute can either be 'calssification' or 'regression'\n",
    "    #split_type attribute can either be 'gini' or 'entropy'\n",
    "    def __init__(self, class_or_reg, split_type='gini', max_depth=2, min_sample_split=1):"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19b5fc65",
   "metadata": {},
   "source": [
    "## Input"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c7fb3220",
   "metadata": {},
   "source": [
    "The input of the of the tree should be two arrays, the first one containing all the features and the second one containing all the labels"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62baaa26",
   "metadata": {},
   "source": [
    "## Outputs"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f206007a",
   "metadata": {},
   "source": [
    "The Decision_tree class had 9 functions, 3 of them are to be used when using the code. The first one is the 'fit' parameter that creates the decision tree but does not return any result. \n",
    "'print_tree' function return the different nodes of the tree with the corresponding splitting values and (only for leaf nodes) the label associated.\n",
    "The last useful function if 'predict' which takes as input the feature for which we want to have the label and returns a series of labels, one for each element to be predicted."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b90b348",
   "metadata": {},
   "source": [
    "## Choose the Best Split\n",
    "We defined functions to calculate gini, entropy for classification, and loss for regression, which are used to decide the best splits. \n",
    "\n",
    "The formulas are the following:\n",
    "\n",
    "$$Square Error = \\sum_{i=1}^{n}(y_{i}-c_{i})^2$$\n",
    "\n",
    "$$Gini = 1-\\sum_{i=1}^{n}p_{i}^{2}$$\n",
    "\n",
    "$$Entropy = \\sum_{i=1}^{n}-p_{i}*log_{2}p_{i}\\$$\n",
    "\n",
    "*where n is the number of splits, c{i} is the mean of all the values in that split, p is the probability\n",
    "\n",
    "Information Gain = Entropy Before the Split - Weighted Average Entropy After the Split\n",
    "If we want larger, information gain, we just need the smaller weighted average entropy. Therefore, to make the algorithm simpler, we focus on achieving smaller square error, gini, and not use information gain."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc01251d",
   "metadata": {},
   "source": [
    "## Build the Tree"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62bbf69a",
   "metadata": {},
   "source": [
    "The dataset is splitted into left and right subsets each step based on one feature value. When the value is smaller than or equal to the split point, it is assigned to left subset, otherwise, it is assigned to the right subset.\n",
    "\n",
    "Then we loop all the features and calculate their corresponding gini, entropy or square error to decide which feature value is best for splitting. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56776e7d",
   "metadata": {},
   "source": [
    "We build the tree recursively. At first, the current depth is set as zero and after each splitting it is increased by one. The minimun number of samples in one node is set as default to 1."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "578e5cad",
   "metadata": {},
   "source": [
    "## Print the Tree"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "02c9556f",
   "metadata": {},
   "source": [
    "In order to provide a visual representation of the tree to understand the splits better, we've implemented a function to visualize the tree named 'print_tree'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e66289b",
   "metadata": {},
   "source": [
    "## Import "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "a46bd359",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import standard packages\n",
    "import numpy as np\n",
    "from sklearn.datasets import make_moons\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0bf60df3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import decision tree\n",
    "import Decision_Tree as dt_scratch"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "390b07e6",
   "metadata": {},
   "source": [
    "## Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6f412119",
   "metadata": {},
   "outputs": [],
   "source": [
    "#for classification\n",
    "x, y = make_moons(100) \n",
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "66634aaa",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "feature_1 <= -0.04553\n",
      "   left: [1.]\n",
      "   right: feature_1 <= 0.5\n",
      "     left: feature_0 <= -0.87132\n",
      "         left: [0.]\n",
      "         right: [1.]\n",
      "     right: [0.]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[array([0.]),\n",
       " array([0.]),\n",
       " array([0.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([0.]),\n",
       " array([0.]),\n",
       " array([0.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([0.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([0.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([1.]),\n",
       " array([1.])]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt = dt_scratch.Decision_tree(class_or_reg='classification', split_type='entropy', max_depth=2)\n",
    "dt.fit(x_train, y_train)\n",
    "dt.print_tree()\n",
    "dt.predict(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e293b23",
   "metadata": {},
   "source": [
    "## Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "54c29329",
   "metadata": {},
   "outputs": [],
   "source": [
    "#for regression\n",
    "N = 100\n",
    "X = np.linspace(-1, 1, N)\n",
    "Y = X**2 + np.random.normal(0, 0.07, N)\n",
    "X = X.reshape(-1, 1)\n",
    "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "4868219d",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "feature_0 <= -0.65657\n",
      "   left: feature_0 <= -0.81818\n",
      "     left: feature_0 <= -0.93939\n",
      "         left: feature_0 <= -1.0\n",
      "                 left: [1.03261546]\n",
      "                 right: [0.89568674]\n",
      "         right: feature_0 <= -0.91919\n",
      "                 left: [0.85663792]\n",
      "                 right: feature_0 <= -0.87879\n",
      "                                 left: [0.77259058]\n",
      "                                 right: [0.81729987]\n",
      "     right: feature_0 <= -0.71717\n",
      "         left: feature_0 <= -0.77778\n",
      "                 left: [0.73045384]\n",
      "                 right: [0.63946665]\n",
      "         right: [0.48547129]\n",
      "   right: feature_0 <= 0.65657\n",
      "     left: feature_0 <= -0.43434\n",
      "         left: feature_0 <= -0.59596\n",
      "                 left: feature_0 <= -0.63636\n",
      "                                 left: [0.48123234]\n",
      "                                 right: [0.34453883]\n",
      "                 right: feature_0 <= -0.53535\n",
      "                                 left: [0.3188669]\n",
      "                                 right: feature_0 <= -0.45455\n",
      "                                                                 left: [0.21602239]\n",
      "                                                                 right: [0.30241409]\n",
      "         right: feature_0 <= 0.43434\n",
      "                 left: feature_0 <= 0.27273\n",
      "                                 left: feature_0 <= -0.27273\n",
      "                                                                 left: feature_0 <= -0.29293\n",
      "                                                                                                                                 left: feature_0 <= -0.31313\n",
      "                                                                                                                                                                                                                                                                 left: feature_0 <= -0.41414\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [0.10623319]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: feature_0 <= -0.35354\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [0.16569719]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: [0.11333386]\n",
      "                                                                                                                                                                                                                                                                 right: [-0.00277186]\n",
      "                                                                                                                                 right: [0.22952271]\n",
      "                                                                 right: feature_0 <= -0.0101\n",
      "                                                                                                                                 left: feature_0 <= -0.0303\n",
      "                                                                                                                                                                                                                                                                 left: feature_0 <= -0.23232\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [0.0788919]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: feature_0 <= -0.17172\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [-0.13104345]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: [-0.03088847]\n",
      "                                                                                                                                                                                                                                                                 right: [-0.11317958]\n",
      "                                                                                                                                 right: feature_0 <= 0.0101\n",
      "                                                                                                                                                                                                                                                                 left: [0.17126278]\n",
      "                                                                                                                                                                                                                                                                 right: feature_0 <= 0.17172\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: feature_0 <= 0.15152\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [0.0158136]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: [0.02603359]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: feature_0 <= 0.21212\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [0.09466722]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: [-0.01637753]\n",
      "                                 right: feature_0 <= 0.31313\n",
      "                                                                 left: [0.19874871]\n",
      "                                                                 right: feature_0 <= 0.33333\n",
      "                                                                                                                                 left: [-0.01008439]\n",
      "                                                                                                                                 right: feature_0 <= 0.37374\n",
      "                                                                                                                                                                                                                                                                 left: [0.18408312]\n",
      "                                                                                                                                                                                                                                                                 right: feature_0 <= 0.41414\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 left: [0.12176561]\n",
      "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 right: [0.17400803]\n",
      "                 right: feature_0 <= 0.59596\n",
      "                                 left: feature_0 <= 0.45455\n",
      "                                                                 left: [0.19743116]\n",
      "                                                                 right: feature_0 <= 0.53535\n",
      "                                                                                                                                 left: [0.26188254]\n",
      "                                                                                                                                 right: [0.28334019]\n",
      "                                 right: feature_0 <= 0.61616\n",
      "                                                                 left: [0.35055099]\n",
      "                                                                 right: [0.41929872]\n",
      "     right: feature_0 <= 0.77778\n",
      "         left: feature_0 <= 0.71717\n",
      "                 left: feature_0 <= 0.69697\n",
      "                                 left: [0.51245092]\n",
      "                                 right: [0.38504335]\n",
      "                 right: [0.63036618]\n",
      "         right: feature_0 <= 0.89899\n",
      "                 left: feature_0 <= 0.81818\n",
      "                                 left: [0.71367579]\n",
      "                                 right: [0.79979401]\n",
      "                 right: feature_0 <= 0.9596\n",
      "                                 left: [0.88151784]\n",
      "                                 right: [0.94586463]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3372: RuntimeWarning: Mean of empty slice.\n",
      "  return _methods._mean(a, axis=axis, dtype=dtype,\n",
      "/opt/anaconda3/lib/python3.8/site-packages/numpy/core/_methods.py:170: RuntimeWarning: invalid value encountered in double_scalars\n",
      "  ret = ret.dtype.type(ret / rcount)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[array([0.21602239]),\n",
       " array([0.63036618]),\n",
       " array([0.89568674]),\n",
       " array([0.0158136]),\n",
       " array([0.26188254]),\n",
       " array([0.79979401]),\n",
       " array([0.48547129]),\n",
       " array([0.26188254]),\n",
       " array([0.0158136]),\n",
       " array([0.88151784]),\n",
       " array([0.0158136]),\n",
       " array([0.79979401]),\n",
       " array([0.28334019]),\n",
       " array([0.63946665]),\n",
       " array([0.81729987]),\n",
       " array([0.16569719]),\n",
       " array([0.3188669]),\n",
       " array([0.21602239]),\n",
       " array([0.11333386]),\n",
       " array([0.81729987])]"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dt2 = dt_scratch.Decision_tree(class_or_reg='regression', max_depth=9)\n",
    "dt2.fit(X_train, Y_train)\n",
    "dt2.print_tree()\n",
    "dt2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47b8d17d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
