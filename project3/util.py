
'''
Author: Dean Hickman
Purpose: useful functions for P3'''

import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.metrics import roc_auc_score
import jax.numpy as jnp
import jax as random


def plot_explanations(NN, X_test, y_test, model_names= None):
    if model_names is None:
        model_names = [f"Model {i}" for i in range(len(NN))]

    for i, model in enumerate(NN):
        model_explanations = []
        while True:
            randomIndex = np.random.choice(len(X_test))
            x_test_instance = X_test[randomIndex]
            y_true = np.argmax(y_test[randomIndex])

            x_test_instance_reshaped = np.expand_dims(x_test_instance, axis=0)

            predicted_label = model.predict(x_test_instance_reshaped)
            predicted_probabilities = model.predict_proba(x_test_instance_reshaped)
            if np.max(predicted_probabilities) < 0.90:
                break

 
        num_features = X_test.shape[1]
        side_length = int(np.sqrt(num_features))

        if side_length * side_length != num_features:
            raise ValueError("Cannot reshape.")

        image = np.reshape(x_test_instance, (side_length, side_length))
        input_gradients = model.input_gradient(x_test_instance, predicted_label)
        abs_input_gradient = np.abs(input_gradients)
        scaled_abs= np.abs(input_gradients * x_test_instance)
        smoothed_gradients = model.smoothed_input_gradient(x_test_instance, predicted_label)
        abs_smoothed_gradients = np.abs(smoothed_gradients)

        model_explanations.append(abs_input_gradient)

        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        #plot 1: image
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title('Original Image')
        axs[0].axis('off') 

        #plot 2: absolute value of the input gradients
        axs[1].imshow(np.reshape(abs_input_gradient, (side_length, side_length)), cmap='hot')
        axs[1].set_title('|Input Gradients|')
        axs[1].axis('off')

        #plot 3: scaled absolutes
        axs[2].imshow(np.reshape(scaled_abs, (side_length, side_length)), cmap='hot')
        axs[2].set_title('Scaled absolutes')
        axs[2].axis('off')

        #plot 4: absolute value of smoothed input gradients
        axs[3].imshow(np.reshape(abs_smoothed_gradients, (side_length, side_length)), cmap='hot')
        axs[3].set_title('|Smoothed Gradients|')
        axs[3].axis('off')
        
        fig.suptitle(f'True label: {y_true} Predicted label: {predicted_label}', fontsize=16)
        plt.show()

def compute_auc_score(NN, X_test, y_test):
    y_pred_proba = NN.predict_proba(X_test) 
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    return auc_score

	
	
    
