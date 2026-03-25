''' Filename: active_learning.py
Author: Dean Hickman
Date: 12/11/2024
Description: This file defines the functions used in P5_Active_Learning.ipynb
'''


import random
import numpy as np 

def compute_random_criterion( unlabeled_indices ):
	'''Computes a random ordering of the unlabeled_indices
	Parameters:
	----------
	unlabeled_indices: list.  
		Indices in X that have not yet been labeled.
	Returns:
	----------
	scores: list. Length = length of unlabeled_indices
		a random ordering of the the integers between 1 and the number of unlabeled_indices
	'''
	scores = random.sample(range(1, len(unlabeled_indices) + 1), len(unlabeled_indices))
	return scores


def compute_variance_criterion(X, unlabeled_indices, model):
    '''Computes the variance criterion on the unlabeled instances in X
    Parameters:
    ----------
    X: ndarray. Shape = [Num samples N, Num features D]
        Collection of input vectors.
    unlabeled_indices: list.  
        Indices in X that have not yet been labeled.
    model: Fitted scikit-learn LogisticRegression model.  (Note: other models with the same predict_proba and fit method signatures will work too)
        Model fit on the labeled_indices in X
    Returns:
    ----------
    scores: list. Length = length of unlabeled_indices
        the variance scores computed for each instance in unlabeled_indices
    '''
    scores = []
    for i in unlabeled_indices:
        probas = model.predict_proba(X[i, :].reshape(1, -1))
        scores.append(np.var(probas))
    return scores

def compute_expected_improvement_criterion( X , y , labeled_indices , unlabeled_indices , model ):
	'''Computes the expected improvement criterion on the unlabeled instances in X
	Parameters:
	----------
	X: ndarray. Shape = [Num samples N, Num features D]
		Collection of input vectors.
	y: ndarray. Shape = [Num samples N,]
		True classes corresponding to each input sample (coded as 0, 1, 2 etc).  Note that some datasets have more than 2 classes.
	labeled_indices: list.  
		Indices in X that have already been labeled.	
	unlabeled_indices: list.  
		Indices in X that have not yet been labeled.
	model: Fitted scikit-learn LogisticRegression model.  (Note: other models with the same predict_proba and fit method signatures will work too)
		Model fit on the labeled_indices in X
	Returns:
	----------
	scores: list. Length = length of unlabeled_indices
		the expected improvement scores computed for each instance in unlabeled_indices
	NOTES:
	1. You'll need to keep a copy of the predicted probabilities for X before the start of your for loops 
		(you'll retrain model inside your for loops.)
	2. Inside the for outer for loop over unlabeled_indices, you'll need a temporary version of labeled_indices 
		and unlabeled_indices that corresponds to the original labeled_indices + the index you're considering 
		(and unlabeled_indices should be the original list - that index).  
	3. Inside the inner for loop over class labels, you'll need a temporary version of the labels where you set 
		the label for the index you're considering to the current class label you're considering.
	'''
	scores = []
	Probas_0 = model.predict_proba(X) #assuming that the model has already been fit to the data

	for i in unlabeled_indices:
		tempLabeled_indices = labeled_indices + [i]
		tempUnlabeled_indices= [idx for idx in unlabeled_indices if idx != 1]

		improvement= 0
		for c in np.unique(y):
			temp_y = y.copy()
			temp_y[i] = c
			#retrain the model
			model.fit(X[tempLabeled_indices], temp_y[tempLabeled_indices])
			new_probabilities = model.predict_proba(X[tempUnlabeled_indices])
			minProbs = np.min(new_probabilities, axis= 1)
			term= np.sum(1- minProbs)
			improvement+= Probas_0[1, int(c)]*term
		scores.append(improvement)

	return scores

def label_next_instance( X , y , labeled_indices , unlabeled_indices , model , criterion ):
	'''Selects the next instance to label according to the specified criterion and moves it from 
		unlabeled_indices to labeled_indices
	Parameters:
	----------
	X: ndarray. Shape = [Num samples N, Num features D]
		Collection of input vectors.
	y: ndarray. Shape = [Num samples N,]
		True classes corresponding to each input sample (coded as 0, 1, 2 etc).  Note that some datasets have more than 2 classes.
	labeled_indices: list.  
		Indices in X that have already been labeled.	
	unlabeled_indices: list.  
		Indices in X that have not yet been labeled.
	model: Fitted scikit-learn LogisticRegression model.  (Note: other models with the same predict_proba and fit method signatures will work too)
		Model fit on the labeled_indices in X
	criterion: String.
		Should be either "variance" or "expected_improvement" and dictates the criterion used to choose the next instance.
	Returns:
	----------
	new_labeled_indices: list.  
		labeled_indices plus the index of the next instance chosen to be labeled	
	new_unlabeled_indices: list.  
		unlabeled_indices minus the index of the next instance chosen to be labeled
	NOTES:
	1. Call either compute_variance_criterion or compute_expected_improvement_criterion depending on criterion
	'''
	if criterion == 'variance':
		scores = compute_variance_criterion(X, unlabeled_indices, model)
	elif criterion == 'expected_improvement':
		scores = compute_expected_improvement_criterion(X, y, labeled_indices, unlabeled_indices, model)
	elif criterion == 'random':
		scores = compute_random_criterion(unlabeled_indices)
	else:
		raise Exception('Should be one of "variance", "expected_improvement", or "random"')

		
	next_instance = unlabeled_indices[np.argmax(scores)]

	new_labeled_indices = labeled_indices + [next_instance]
	new_unlabeled_indices = [idx for idx in unlabeled_indices if idx != next_instance]
	
	return new_labeled_indices, new_unlabeled_indices

def initialize_labeled_indices(y):
    '''Initialized a labeled_indices list by randomly selecting an index corresponding to an instance from each class
        and the corresponding unlabeled_indices list
    Parameters:
    ----------
    y: ndarray. Shape = [Num samples N,]
        True classes corresponding to each input sample (coded as 0, 1, 2 etc).  Note that some datasets have more than 2 classes.
    Returns:
    ----------
    labeled_indices: list.  
        n_classes randomly chosen indices, each corresponding to a different class label in y    
    unlabeled_indices: list.  
        All indices corresponding to instances in y that are not in labeled_indices
    '''
    labeled_indices = []
    unlabeled_indices = []

    classes = np.unique(y)
    for i in classes:
        indices = np.where(y == i)[0]
        random_index = np.random.choice(indices)
        labeled_indices.append(random_index)
    
    for j in range(len(y)):
      if j not in labeled_indices:
        unlabeled_indices.append(j)
        
    return labeled_indices, unlabeled_indices

