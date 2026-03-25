'''TODO:
Author: Dean Hickman
This is my implementation of the LogisticRegression class'''

##Keep these imports as-is
import jax.numpy as jnp
from jax import grad
from jax.nn import sigmoid
from jax import random
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
	''' Logistic regression with option to add interaction terms with their own sparsity penalty.'''

	##NOTE: penalty and alpha are used starting at step 2, interactions starting at step 3 and interaction_alpha starting at step 4
	def __init__( self , feature_names , step_size=1e-2 , n_iter=5000 , penalty=None , alpha=0. , interactions=False , interaction_alpha=None ):

		##NOTE: Each of these comments corresponds to a class field that you are suggested to have.  If there's no step number, you'll need them from the first step!

		##STEP 2 -- Type of penalty.  Options are None, 'l1', 'l2'

		##STEP 2 -- Penalty weight

		##STEP 3 -- Flag for including interaction terms

		##STEP 3 -- Standard scaler - to use for scaling interaction terms

		##Feature names -- (STEP 3: these should include interaction terms where applicable)

		##STEP 4 -- Indices of non-interaction features

		##STEP 4 -- Indices of interaction features (empty if none)

		##STEP 4 -- Interaction terms penalty weight.  If none, set to alpha.

		##Number of features (STEP 3: This should include interaction terms where applicable)

		##Number of iterations of gradient descent to use in training

		##Step size to use when updating parameters in gradient descent

		##Randomly initialized weights of size self.n_features with mean 0 and variance 1/100.

		##Randomly initialized bias of size 1, with mean 0 and variance 1/100.
		self.feature_names= feature_names
		self.step_size= step_size
		self.n_iter= n_iter
		self.penalty= penalty 
		self.alpha= alpha
		self.interactions= interactions
		self.interaction_alpha= interaction_alpha if interaction_alpha is not None else alpha

		

	def process_features( self , X ):
		'''STEP 3
		Add interaction terms and apply standard scaling to the processed feature vector (to scale the interaction terms)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.

		Returns:
		----------
		A processed collection of input vectors with interaction terms added if self.interactions=True, and with standard scaling applied
		
		NOTES: 
		1. Only fit self.Scaler on the training set.  You can do this by only fitting it if self.scaler is None (that way you only fit it once).
		'''


	def predict_proba_with_params( self , params , X ):
		'''Predicts the probability of class=1 corresponding to each row of X (each instance)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		params: tuple.
			Tuple of ( weights , biases ) where weights is ndarray and has shape = [Num features D,] 
			and biases is ndarray and has shape = [1,] 

		Returns:
		----------
		1-D array of predicted probabilities of class=1 for each input feature vector. Shape = [Num samples N,]
		'''
		weights, bias= params
		X= self.process_features(X)
		z= jnp.dot(X, weights)+bias
		return sigmoid(z)
		## STEP 3: Call processed_features on X and use that as the input to your model instead of just X
	


	def predict_proba( self , X ):
		'''Predicts the probability of class=1 corresponding to each row of X (each instance)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.

		Returns:
		----------
		1-D array of predicted probabilities of class=1 for each input feature vector. Shape = [Num samples N,]

		NOTES: 
		1. This function should call and return the output of predict_proba_with_params using the class's parameters (weights and biases)
		'''

		return self.predict_proba_with_params((self.weights, self.bias), X)


	


	def predict( self , X ):
		'''Predicts the class corresponding to each row of X (each instance)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.

		Returns:
		----------
		1-D array of predicted classes (0 or 1) for each input feature vector. Shape = [Num samples N,]

		NOTES: 
		1. Consider calling your predict_proba function.
		2. Make sure your code doesn't have for loops.  You could use the round function in jax.numpy
		'''
		y_proba= self.predict_proba(X)
		return jnp.round(y_proba)


	def fit( self , X , y ):
		'''Fits weights and bias based on features X and labels y.

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		y: ndarray. Shape = [Num samples N,]
			True classes corresponding to each input sample (coded as 0 or 1).

		Returns:
		----------
		Nothing

		NOTES: 
		1. Call the update function n_iter times and use it to update the weights and biases
		'''
		for i in range (self.n_iter):
			self.weights, self.bias = self.update((self.weights, self.bias), X, y)
			


	def binary_cross_entropy_loss( self , y , y_proba ):
		'''Computes and returns the binary cross entropy loss

		Parameters:
		----------
		y: ndarray. Shape = [Num samples N,]
			True classes corresponding to each input sample (coded as 0 or 1).
		y_proba: ndarray. Shape = [Num samples N,]
			Predicted probabilities of class=1 corresponding to each input sample
 

		Returns:
		----------
		The binary cross entropy loss

		NOTES: 
		1. Don't use a for loop!
		'''
		return -jnp.sum(y * jnp.log(y_proba) + (1 - y)* jnp.log(1- y_proba))

	def sparsity_loss( self , w ):
		'''STEP 2
		Computes and returns the sparsity loss

		Parameters:
		----------
		w: ndarray. Shape = [Num features D,] 
  
		Returns:
		----------
		The sparsity loss

		NOTES: 
		1. Use Jax.numpy's linalg.norm function.  Set ord=1 for an l1 penalty, and ord=2 for an l2 penalty.  
		'''

		##STEP 4: Compute the norm of the weights corresponding to the basic (non-interaction) and weight it by alpha,
		## 		  then compute the norm of the weights corresponding to the interaction terms and weight it by interaction_alpha
		##		  These 2 terms should be added together for the sparsity loss


	##Need updating
	def loss( self , params , X , y ):
		'''Fits weights and bias based on features X and labels y.

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		y: ndarray. Shape = [Num samples N,]
			True classes corresponding to each input sample (coded as 0 or 1).
		params: tuple.
			Tuple of ( weights , biases ) where weights is ndarray and has shape = [Num features D,] 
			and biases is ndarray and has shape = [1,] 

		Returns:
		----------
		The binary cross entropy loss + the sparsity loss (after step 2)

		NOTES: 
		1. Use your predict_proba_with_params function
		2. Clip the predicted probas using Jax.numpy's clip function to be within (1e-14, 1 - 1e-14) so you don't get NaNs when you take the log
		'''
		y_proba= self.predict_proba_with_params(params, X)
		y_proba= jnp.clip(y_proba, 1e-14, 1- 1e-14)
		binaryLoss= self.binary_cross_entropy_loss(y, y_proba)
		sparsityLoss= self.alpha * self.sparsity_loss(params[0])
		return binaryLoss + sparsityLoss


	def update( self , params , X , y ):
		'''Updates and returns the weights and biases

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		y: ndarray. Shape = [Num samples N,]
			True classes corresponding to each input sample  (coded as 0 or 1).
		params: tuple.
			Tuple of ( weights , biases ) where weights is ndarray and has shape = [Num features D,] 
			and biases is ndarray and has shape = [1,] 

		Returns:
		----------
		The updated weights, the updated bias

		NOTES: 
		1. Use jax's grad function to take the gradient of your loss function
		2. Remember to use the step_size when updating the parameters
		'''
		gradient = grad(self.loss)(params, X, y)

		weight_grad, bias_grad= gradient 

		weights= params[0] - self.step_size * weight_grad
		biases= params[1] - self.step_size * bias_grad
		return weights, biases



