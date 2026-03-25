'''
Author: Dean Hickman
Date: 12-17-24
Purpose: CS348 for Project 3'''

##Keep these imports as-is
import jax.numpy as jnp
from jax import grad
from jax.nn import sigmoid , one_hot , relu , softmax
from jax import random

class NeuralNetwork:
    '''Multi-layer Perceptron neural network trained with stochastic gradient descent'''

    def __init__( self , n_features , n_classes , layer_sizes , step_size=1e-2 , n_iter=10000 , batch_size=200 , alpha=0. ):
            self.n_features = n_features 
            self.n_classes = n_classes
            self.layer_sizes = layer_sizes
            self.step_size = step_size
            self.n_iter = n_iter
            self.batch_size = batch_size 
            self.alpha = alpha 
            self.key = random.PRNGKey(0) 
             
            self.params = []
            key = self.key
            for i in range(1, len(layer_sizes)):
                key, subkey = random.split(key)
                weights = random.normal(subkey, (layer_sizes[i - 1], layer_sizes[i])) * jnp.sqrt(2 / (layer_sizes[i - 1] + layer_sizes[i]))
                biases = jnp.zeros((layer_sizes[i],)) 
                self.params.append((weights, biases))
                
    def load_parameters(self, model):
        '''
        Load pre-trained weights & biases into neural network
        
        Parameters:
			- model: Dictionary that contains 'Weights' & 'Biases'
        '''
        self.params = [(jnp.array(w), jnp.array(b)) for w, b in zip(model['weights'], model['biases'])]
 

    def predict_proba_with_params( self , params , X ):
        '''Predicts the probability of of each class for each row of X (each instance)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		params: list.
			List of tuples of ( weights_i , biases_i ) where weights_i is ndarray and has shape = [layer_sizes[i-1], layer_sizes[i]]  
			Note: layer 0 has size D and the last layer has size n_classes. biases is ndarray and has shape = layer_sizes[i] 

		Returns:
		----------
		ndarray of predicted probabilities of each of C classes for each input feature vector. Shape = [Num samples N, Num classes C]
		'''
        activation = X
        
        for i , (weights, biases) in enumerate(params[:-1]):
            activation = jnp.dot(activation, weights) + biases 
            
            if i < len(params) -2: 
                activation = relu(activation)
                
        Weights, Biases = params[-1] 
        logits = jnp.dot(activation, Weights) + Biases
        return softmax(logits, axis = -1)
        


    def predict_proba( self , X ):
        '''Predicts the probability of of each class for each row of X (each instance)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.

		Returns:
		----------
		ndarray of predicted probabilities of each of C classes for each input feature vector. Shape = [Num samples N, Num classes C]
		'''
        return self.predict_proba_with_params(self.params, X)


    def predict( self , X ):
        '''Predicts the class corresponding to each row of X (each instance)

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.

		Returns:
		----------
		1-D array of predicted classes (0, ..., C-1) for each input feature vector. Shape = [Num samples N,]
		'''
        prob = self.predict_proba(X) 
        if len(prob.shape) == 1:
            return jnp.argmax(prob)
        return jnp.argmax(prob, axis = 1)


    def fit( self , X , y ):
        '''Fits weights and bias based on features X and labels y.

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		y: ndarray. Shape = [Num samples N, Num classes C]
			True classes corresponding to a onehot encoding of the class for each input sample

		Returns:
		----------
		Nothing
		'''
        X = X.astype(jnp.float32)
        y = y.astype(jnp.float32)

        for i in range(self.n_iter):
            self.key, subkey = random.split(self.key) #stochastic gradient descent
            batch_indices = random.choice(subkey, X.shape[0], shape=(self.batch_size,), replace=False)
            X_batch = X[batch_indices] #samples
            y_batch = y[batch_indices] #labels
            grads = grad(self.loss)(self.params, X_batch, y_batch)
            self.params = self.update(self.params, grads)
            
            #adjust the learning rate every 500 iterations
            if i % 500 == 0 and i > 1000:
                 self.step_size /= 2
            

    def cross_entropy_loss( self , y , y_proba ):
        '''Computes and returns the multiclass cross entropy loss

		Parameters:
		----------
		y: ndarray. Shape = [Num samples N, Num classes C]
			True classes corresponding to a onehot encoding of the class for each input sample
		y_proba: ndarray. Shape = [Num samples N, Num classes C]
			Predicted probabilities of each class corresponding to each input sample
 

		Returns:
		----------
		The multiclass cross entropy loss
		'''
        y_proba = jnp.clip(y_proba, 1e-8, 1 - 1e-8)
        return -jnp.sum(y * jnp.log(y_proba)) / y.shape[0] #cross-entropy loss function

    def l2_regularization_loss( self, params ):
        '''
		Computes and returns the l2 norm of all of all of the weights across all layers

		Parameters:
		----------
		params: list.
			List of tuples of ( weights_i , biases_i ) where weights_i is ndarray and has shape = [layer_sizes[i-1], layer_sizes[i]]  
			Note: layer 0 has size D and the last layer has size n_classes. biases is ndarray and has shape = layer_sizes[i] 

		Returns:
		----------
		The l2 penalty of all of the weights
		'''
        return sum(jnp.sum(weights**2) for weights, _ in params)

    def loss( self , params , X , y ):
        '''Fits all of the weights and bias based on features X and labels y.

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		y: ndarray. Shape = [Num samples N, Num classes C]
			True classes corresponding to a onehot encoding of the class for each input sample
		params: list.
			List of tuples of ( weights_i , biases_i ) where weights_i is ndarray and has shape = [layer_sizes[i-1], layer_sizes[i]]  
			Note: layer 0 has size D and the last layer has size n_classes. biases is ndarray and has shape = layer_sizes[i] 


		Returns:
		----------
		The cross entropy loss + alpha * the l2 regularization loss

		NOTES: 
		1. Use your predict_proba_with_params function
		2. Clip the predicted probas using Jax.numpy's clip function to be within (1e-5, 1 - 1e-5) so you don't get NaNs when you take the log
		'''
        y_proba = self.predict_proba_with_params(params, X)
        cross_entropy = self.cross_entropy_loss(y, y_proba)
        return cross_entropy + self.l2_regularization_loss(params) + self.alpha

    def update( self , params , grads):
        '''Updates and returns the params

		Parameters:
		----------
		X: ndarray. Shape = [Num samples N, Num features D]
			Collection of input vectors.
		y: ndarray. Shape = [Num samples N, Num classes C]
			True classes corresponding to a onehot encoding of the class for each input sample
		params: list.
			List of tuples of ( weights_i , biases_i ) where weights_i is ndarray and has shape = [layer_sizes[i-1], layer_sizes[i]]  
			Note: layer 0 has size D and the last layer has size n_classes. biases is ndarray and has shape = layer_sizes[i] 

		Returns:
		----------
		The updated params

		NOTES:
		1. Use stochastic gradient descent this time to compute the gradient on a 
			random batch of self.batch_size instances in each iteration
		2. Anneal self.step_size by dividing it by 2 every 1000 iterations
		'''
        newParams = []
        
        for (weights, biases), (grad_w, grad_b) in zip(params, grads):
            newWeights = weights - self.step_size * grad_w
            newBiases = biases - self.step_size * grad_b
            newParams.append((newWeights, newBiases))
        return newParams


    def input_gradient( self , x , c ):
        '''Computes the input gradient explanation of instance x for class x

		Parameters:
		----------
		x: ndarray. Shape = [Num features D,]
			A single input vector.
		c: int.
			Class for which to generation an explanation

		Returns:
		----------
		The input gradient explanation of instance x for class c
		'''
        x = x.astype(jnp.float32)
        
        #Class 'c'
        def loss_for_input(x):
            proba = self.predict_proba_with_params(self.params, jnp.expand_dims(x, axis=0))
            scalar_loss = -jnp.squeeze(proba[0, c])
            return scalar_loss
        return grad(loss_for_input)(x)

    def smoothed_input_gradient( self , x , c, num_samples = 50, noise_scale= 0.1):
        '''Computes the smoothed input gradient explanation of instance x for class x

		Parameters:
		----------
		x: ndarray. Shape = [Num features D,]
			A single input vector.
		c: int.
			Class for which to generation an explanation

		Returns:
		----------
		The smoothed input gradient explanation of instance x for class c
		'''
        gradientSum = jnp.zeros_like(x)
        for _ in range(num_samples):
            noise = random.normal(self.key, shape=x.shape) * noise_scale
            gradientSum += self.input_gradient(x + noise, c)
        return gradientSum / num_samples     
    def randomized_layer(self, layer_index):
        key, subkey = random.split(self.key)
        weights_shape = self.params[layer_index][0].shape 
        bias_shape = self.params[layer_index][1].shape 
        update_weights = random.normal(subkey, weights_shape) * jnp.sqrt(2/ sum(weights_shape))
        update_biases = jnp.zeros(bias_shape)
        self.params[layer_index] = (update_weights, update_biases)
    def evaluate(self, X_test, y_test):
        X_test = X_test.astype(jnp.float32)
        y_test = y_test.astype(jnp.float32)
        y_pred = self.predict(X_test)
        if len(y_test.shape) > 1:
            y_test = jnp.argmax(y_test, axis=1)
        accuracy = jnp.mean(y_pred == y_test)*100
        print(f"Accuracy: {accuracy: .2f}%")
        return accuracy

			