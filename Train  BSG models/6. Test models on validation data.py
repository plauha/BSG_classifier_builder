### This notebook shows how the performance of the classification models was assessed in article "Bird Sounds Global - model builder: An end-to-end workflow for building locally fine-tuned bird classifiers"

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from classifier import Classifier
from functions import top_preds
import os
import gc
import sys
import numpy as np
import shutil
from sklearn import metrics

# Check results of model training

model_name = 'models/Spain_Portugal/'
rounds = 2 # in how many runs was the model trained

# visualize results of network training
def plot_results(history, val = True):
    acc = history['binary_accuracy']
    loss = history['loss']
    if val:
        val_acc = history['val_binary_accuracy']
        val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    if val:
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training (and validation) accuracy')
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    if val:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training (and validation) loss')
    plt.legend()
    plt.grid()
    plt.show()

joint_history = {'loss':[], 'binary_accuracy':[], 'val_loss':[], 'val_binary_accuracy':[]}

for i in range(rounds):
    with open(model_name + 'training_history_' + str(i+1) + '.pkl', 'rb') as f:
        history = pickle.load(f)
        for stat in ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy']:
            joint_history[stat] = joint_history[stat] + history[stat]

plot_results(joint_history)

# Analyze validation data

# analyze validation data with own model
model_version = 'model_v1.2.tflite'

class_list = pd.read_csv(model_name + 'classes.csv')
path_to_data = model_name + 'local_training_data/'
os.mkdir('validation_data')

with open(path_to_data + 'metadata/val_set.pkl', 'rb') as f:
    val_data = pickle.load(f)
with open(path_to_data + 'metadata/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

val_data = [f for f in val_data if f[-11:]!='cleaned.wav'] # only analyze clips with original background 
for f in val_data:
    shutil.move(path_to_data + f, 'validation_data/' + f) 
    
cls = Classifier(path_to_model=model_name + model_version)
final_preds = np.zeros((len(val_data), len(class_list)))

for i in range(len(val_data)):
    pred, t = cls.classify('validation_data/' + val_data[i], max_pred=False)
    pred = pred[0,:] # select first 3 seconds
    
    with open('test_results.txt', 'a') as f:
        f.write(str(val_data[i]))
        for p in pred:
            f.write(',')
            f.write(str(p))
        f.write('\n')

test_results = pd.read_csv('test_results.txt', header=None)

# create a data frame of true labels

true_lab = pd.DataFrame(0, index=np.arange(len(val_data)), columns=list(range(len(class_list)+1)))
true_lab[0] = val_data

for i in range(len(true_lab)):
    true_lab.loc[i, 1:len(class_list)+1] = labels[true_lab[0].iloc[i]].astype(int)

true_lab.to_csv('true_labels.csv', index=False)   

# analyze with BirdNET-Analyzer by running following command on command line
# python3 analyze.py --i validation_data/ --o birdnet_results/ --min_conf 0.01


for f in test_results[0]:
    data = pd.read_csv('birdnet_results/' + f.replace('wav', 'BirdNET.selection.table.txt'), sep = '\t')
    filename = f
 
    with open('BN_test_results.txt', 'a') as f:
        f.write(filename)
        for sp_code in class_list['species_code']:
            f.write(',')
            conf = data['Confidence'].loc[data['Species Code']==sp_code]
            if len(conf)>0:
                f.write(str(conf.iloc[0]))
            else:
                f.write(str(0))
        f.write('\n')

bn_results = pd.read_csv('BN_test_results.txt', header=None)

# Calculate AUCs

ns = np.zeros(len(class_list)-2)

df = pd.DataFrame({'n':ns})

results = ['test_results.txt', 'BN_test_results.txt']
model = ['BSG', 'birdnet']

for k in range(2):
    aucs = np.zeros(len(class_list)-2)
    classes = np.zeros(len(class_list)-2)
    data=pd.read_csv(results[k], sep=',', header=None)
    
    for i in range(3,len(class_list)+1):
        classes[i-3] = i-1
        try:
            aucs[i-3] = metrics.roc_auc_score(true_lab[i], data[i])
            ns[i-3] = sum(true_lab[i]==1)
        except:
            print("Can not process class " + str(i))
    df['n']=ns
    df['class'] = classes
    df[model[k]] = aucs

df['model'] = model_name.split('/')[1]

df

# Show species with at least 5 positive test samples
df.loc[df['n']>5]

# add results to existing results from previous models
aucs = pd.read_csv('all_aucs.csv')
aucs = pd.concat([aucs, df])
aucs.to_csv('all_aucs.csv', index=False)

import seaborn as sns

plt.rcParams["figure.figsize"] = (5,6)

aucs = pd.read_csv('all_aucs.csv')
aucs = aucs.loc[aucs['n']>4]
aucs = aucs.loc[aucs['birdnet']!=0.5] # exclude species not included in BirdNET 
aucs = aucs.loc[aucs['birdnet']!=0]
aucs1 = aucs[['n', 'class', 'BSG', 'birdnet', 'model']]
aucs1['Training data'] = 'Local + global'
aucs2 = aucs[['n', 'class', 'BSG_xc', 'birdnet', 'model']]
aucs2.columns = ['n', 'class', 'BSG', 'birdnet', 'model']
aucs2['Training data'] = 'Global only'
aucs = pd.concat([aucs1, aucs2])
aucs['difference'] = aucs['BSG'] - aucs['birdnet']

sns.boxplot(data=aucs, x='difference', y='model', hue = 'Training data', whis = 2, order=['Central Europe', 'Iberian Peninsula', 'Northern Argentina', 'Madagascar', 'Finland*', 'Mexico*'], color='darkgrey')
plt.xlim(-0.2, 0.5)
plt.vlines(0, -0.5, len(np.unique(aucs['model']))-0.5, colors='dimgrey', linestyles='dashed')
#plt.title('Model comparison BSG - BirdNET')
plt.xlabel('BSG Classifier AUC - BirdNET AUC')
plt.ylabel('Model')
plt.yticks(ticks=range(len(np.unique(aucs['model']))), labels=['Central\nEurope', 'Iberian\nPeninsula', 'Northern\nArgentina', 'Madagascar', 'Finland*', 'Mexico*'])
plt.legend(bbox_to_anchor=(-0.02, 1.00, 1, 0.165), loc="upper left", title ='Training data')

#plt.savefig('model_comparison.tiff', dpi = 600, format='tiff', bbox_inches='tight')
plt.savefig('model_comparison.svg', dpi=600, format='svg', bbox_inches='tight')
plt.savefig('model_comparison.png', bbox_inches='tight')
plt.show()

# Print AUC comparison

aucs = pd.read_csv('all_aucs.csv')
aucs['difference'] = aucs['BSG'] - aucs['birdnet']
aucs = aucs.loc[aucs['n']>4]
aucs = aucs.loc[aucs['birdnet']==0.5] # AUCs for species not included in BirdNET 
print(f"BSG AUC (non-BirdNET species): {round(np.mean(aucs['BSG']),3)}") 
print(f"BSG-xc AUC (non-BirdNET species): {round(np.mean(aucs['BSG_xc']),3)}") 
print()

aucs = pd.read_csv('all_aucs.csv')
aucs['difference'] = aucs['BSG'] - aucs['birdnet']
aucs = aucs.loc[aucs['n']>4]
aucs = aucs.loc[aucs['birdnet']!=0.5] # AUCs for species included in BirdNET 
print(f"BSG AUC (BirdNET species): {round(np.mean(aucs['BSG']),3)}")
print(f"BSG-xc AUC (BirdNET species): {round(np.mean(aucs['BSG_xc']),3)}")
print(f"BirdNET AUC (BirdNET species): {round(np.mean(aucs['birdnet']),3)}")

# print average AUCs per model

aucs = pd.read_csv('all_aucs.csv')
aucs = aucs.loc[aucs['n']>4]

print('BSG')
for m in np.unique(aucs['model']):
    print(f"{m}: {round(np.mean(aucs['BSG'].loc[aucs['model']==m]),3)}, (n={len(aucs.loc[aucs['model']==m])})")

print()
print('BSG_xc')
for m in np.unique(aucs['model']):
    print(f"{m}: {round(np.mean(aucs['BSG_xc'].loc[aucs['model']==m]),3)}, (n={len(aucs.loc[aucs['model']==m])})")
    
aucs = pd.read_csv('all_aucs.csv')
aucs = aucs.loc[aucs['n']>4]
aucs = aucs.loc[aucs['birdnet']!=0.5]

print()
print('BirdNET')
for m in np.unique(aucs['model']):
    print(f"{m}: {round(np.mean(aucs['birdnet'].loc[aucs['model']==m]),3)}, (n={len(aucs.loc[aucs['model']==m])})")
    
# Calculate p-values for difference between BSG / BirdNET AUCs

from scipy import stats

aucs = pd.read_csv('all_aucs.csv')
aucs = aucs.loc[aucs['n']>4]
aucs = aucs.loc[aucs['birdnet']!=0.5]
models = np.unique(aucs['model'])

print('BSG vs BirdNET')
for m in models:
    print(f"{m}: p-value:{round(stats.ttest_rel(aucs['BSG'].loc[aucs['model']==m], aucs['birdnet'].loc[aucs['model']==m]).pvalue, 5)}")
   
print('')
print('BSG-xc vs BirdNET')
for m in models:
    print(f"{m}: p-value:{round(stats.ttest_rel(aucs['BSG_xc'].loc[aucs['model']==m], aucs['birdnet'].loc[aucs['model']==m]).pvalue, 5)}")    
    
aucs = pd.read_csv('all_aucs.csv')
aucs = aucs.loc[aucs['n']>4]
print('')
print('BSG-xc vs BirdNET')
for m in models:
    print(f"{m}: p-value:{round(stats.ttest_rel(aucs['BSG_xc'].loc[aucs['model']==m], aucs['BSG'].loc[aucs['model']==m]).pvalue, 5)}")        