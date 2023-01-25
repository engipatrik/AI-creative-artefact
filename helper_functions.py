# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:04:52 2023

@author: Patrik.Engi
"""

# A function used for normalising the 
def normalise(data,maxi='none',mini='none'):
    feature_length=data.shape[1]
    normalised_data=np.zeros([len(data),1])
    for i in range (0,feature_length+1):
        maximum=np.amax(data[:,i-1:i], axis=0)
        minimum=np.amin(data[:,i-1:i],axis=0)
        feature=data[:,i-1:i]
        normalised_feature=(feature-minimum)/(maximum-minimum)
        normalised_data=np.concatenate((normalised_data,normalised_feature), axis=1)
    return normalised_data[:,1:]


# This function is used for plotting the heatmap of the confusion matrix 
def confusion_heatmap(y_test, y_pred):
    # use sklearn function to generate the matrix
    c_mat = confusion_matrix(y_test, y_pred)
    
    # Set figure and font
    plt.figure(figsize=(8,6))
    sns.set(font_scale=1.2)
    
    # Plot the heatmap 
    sns.heatmap(cm, annot=True, fmt = 'g', cmap="Greens", cbar = False)
    
    # Add labels 
    plt.xlabel("Predicted Class", size = 18)
    plt.ylabel("True Class", size = 18)
    plt.title("Confusion matrix", size = 20)
    plt.show()
    
    
def plot_loss_curves(list1, list2, epochs, config):
    if config == "loss":
        labels = ["Model loss", "Loss"]
    elif config == "acc":
        labels = ["Model Accuracy", "Accuracy"]
    elif config == "auc":
        labels = ["Model AUC", "AUC"]
        
    epochs_range = range(1,epochs+1)
    plt.figure(figsize = (12,6))
    plt.plot(epochs_range,list1)
    plt.plot(epochs_range,list2)
    plt.title(labels[0])
    plt.xlabel('Epoch')
    plt.ylabel(labels[1])
    plt.legend(['Train','Validation'],loc='upper left')
    plt.show()

def plot_auc(y_test, y_pred):
    false_positive_rate, true_positive_rate, thresolds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
def normalise(data,maxi='none',mini='none'):
    feature_length=data.shape[1]
    normalised_data=np.zeros([len(data),1])
    for i in range (0,feature_length+1):
        print(i)
        maximum=np.amax(data[:,i-1:i], axis=0)
        minimum=np.amin(data[:,i-1:i],axis=0)
        feature=data[:,i-1:i]
        normalised_feature=(feature-minimum)/(maximum-minimum)
#        normalised_feature=np.reshape(normalised_feature,(len(normalised_feature),1))
        normalised_data=np.concatenate((normalised_data,normalised_feature), axis=1)
    return normalised_data[:,1:]
