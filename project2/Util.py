#Author: Dean Hickman
#Filename: Util.py
#Purpose: A folder to hold all of the useful functions from projects in CS348


#1. plot model weights
import matplotlib.pyplot as plt
import numpy as np

def plot_model_weights(model, feature_names):
    weights = model.coef_[0]
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(weights)), weights, color= 'blue')
    plt.xticks(np.arange(len(weights)), feature_names, rotation = 90)
    

    plt.ylabel("Feature Weight")
    plt.title("Model Weights")

    plt.show()


#2. lime explanations
import lime 
import lime.lime_tabular
import numpy as np 

def explainer(model, trainData, testData, feature_names):
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data= trainData,
        feature_names=feature_names,
        mode = 'classification',
        verbose= True
    )
    index= np.random.choice(testData.shape[0], 10)
    chosenInstances= testData[index]
    explanations = []
    localFeature=[]
    for i in index:
        explanation= lime_explainer.explain_instance(
            data_row = testData[i],
            predict_fn= model.predict_proba,
            num_features = 5,
            num_samples = 50000
        )
        explanations.append(explanation)
        localFeature.append(explanation.as_list())
    
    for i, features in enumerate(localFeature):
        print("feature importance for instance: ", i)
        for feature, importance in features:
            print(feature, importance)

    return explanations, chosenInstances, localFeature
    
    
        
        
#3 
import numpy as np 
from sklearn.metrics import accuracy_score,roc_auc_score

def print_model_evaluation(model, trainData, y_train, testData, y_test):

    if len(model.predict_proba(trainData).shape)==1:
        model.predict_proba1= model.predict_proba(trainData)
        model.predict_proba1=model.predict_proba(testData)
    else:
        model.predict_proba1= model.predict_proba(trainData)[:, 1]
        model.predict_proba1=model.predict_proba(testData)[:, 1]

    print("The fraction of positive labels in the train set:", np.mean(y_train))
    print("The fraction of positive labels in the test set:", np.mean(y_test))

    print("The accuracy score for the train set:", accuracy_score(y_train, model.predict(trainData)))
    print("The accuracy score for the test set is:", accuracy_score(y_test, model.predict(testData)))

    print("The train AUC score is:", roc_auc_score(y_train, model.predict_proba(trainData)[:,1]))
    print("The test AUC score is:", roc_auc_score(y_test, model.predict_proba(testData)[:,1]))
