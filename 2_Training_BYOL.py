#!/usr/bin/env python3

###### Training ResNet 50 with BYOL and custom Augmentations
### Author: Annalena Weissert 
## Last Edited: 20 July 2026
# Source: Code segments for model training were partially adaptad from Guo, et. al., 2024, "DeepION: A Deep Learning-Based Low-Dimensional Representation Model of Ion Images for Mass Spectrometry Imaging", published under CC-BY 4.0 license. 
#         Code segments for plotting, validation and early stopping are written by me.


## Packages
import os             
import sys
import time          # To track calculation times
import numpy as np
import torch 

from torchvision import models # Assess Resnet model
from byol_pytorch import BYOL  # IMPORTANT: Published Weights use Version 0.8.2. 'pip install byol-pytorch==0.8.2' Source: https://codeberg.org/lucidrains/byol-pytorch.git (Last accessed: 08 July 2026)

import matplotlib.pyplot as plt    # For visualization
from numpy.linalg import norm      # for cosine simlarity 
import sklearn.metrics as metrics  # For ROC Curve

sys.path.append(os.getcwd()) # Add the working directory to the system path
# Import Custom Augmentation functions
from AugmentationCuda import augment_ref   # pyright: ignore[reportMissingImports]
from AugmentationCuda import augment_maldi # pyright: ignore[reportMissingImports]


################! ADJUST THESE VARIABLES ##################
### Depending on the available computational resources and input data, these parameters may be adjusted accordingly.
## The default settings will run on data from "ExampleData_Processed" and "ExampleData_Training" folder, assuming the same folder structure as provided on github.
## These are reduced datasets. To recreate the results from the paper use the Processed Data provided here: https://doi.org/10.17877/RTG2624-2026-MRKM1K1K 
#                              and create the tarining dataset via the script "1_Fata_processed_to training.ipynb"
# 

## Naming ID: Used in the filename that saves the trained weights (together with the current date). ! Important: Change before each run, otherwise results get overwritten
id = 123  #!

## Data Paths

## Path settings for Example Data
processed_data_path = "ExampleData_Processed"  #! Relative path to the folder containing the processed .pt data files
training_data_path = "ExampleData_Training"    #! Relative path to the folder containing the trainig images as .pt file for each image
valdiation_tumors = ["ExampleTumor5.pt","ExampleTumor6.pt"]       #! Define names of validation tumors. Training progress will be tracked on these.
label_paths = ["ExampleData_Labels/Labels_ExampleTumor5.pt","ExampleData_Labels/Labels_ExampleTumor6.pt"]   #! Relative paths to the label files correpsonding to each validation tumor                                                                          
mz_path = "ExampleData_Processed/00mz_names.pt" #! path and name to the .pt file containing the names of all m/z-values

## Path settings for Real Data
#processed_data_path = "Processed_Data"  
#training_data_path = "Training_Data"    
#valdiation_tumors = ['Edi3-Metastase-M1-Lu-Left.pt', 'Primaer-Tumor-M18.pt', 'Rezidiv-Tumor-M22.pt']     
#label_paths = ['Labels/Labels_M1L.pt', 'Labels/Labels_P18.pt', 'Labels/Labels_R22.pt']                                                                             
#mz_path = "Processed_Data/00mz_names.pt"                                                    



## Training Parameters 
epoch_amount = 30 #!! maximum epochs (if it doesn't get stopped early)
mini_batch = 150   # how many images in one batch (it's recommended to use multliples of 8 and the larger the batch size the better) 
#!!mini_batch = 200 

input_size = 232   # Size of the (squared) training images. Here: 224 + padding of 8 




################ VARIABLE INITIALIZATION #########################################
# Today's date
date = time.strftime("%d_%m_%Y",time.localtime(time.time()))

## CNN Model - here ResNet-50
model_name = "ResNet50" # variable used for naming output
model = models.resnet50(weights='DEFAULT').cuda() 
weights = models.ResNet50_Weights.IMAGENET1K_V2              # As for 02 July 2026 this is equivalent to ResNet50_Weights.DEFAULT
feature_amount = list(model.children())[-1:][0].in_features  # Remove last layer as it is specific for classification

## load and process mz value names
mz = torch.load(mz_path, weights_only=False)  
number_mz = len(mz)
mz = np.concatenate((['Ref'],np.round(mz[1:].astype(float),3))) # Keep 'Ref' at index 0 and round the m/z-values to 3 digits 

## List all training image names to sample from
training_tumors = [os.path.join(training_data_path, file) for file in os.listdir(training_data_path)]

## Number of validation tumors
number_val_tumors = len(valdiation_tumors)

## Initialize training variables
epoch_loss = np.zeros(epoch_amount)
rng = np.random.default_rng(1807)   # Set seed for shuffling training data

preprocess = weights.transforms()

## Early stopping parameters
patience = 50        # Number of epochs to wait before stopping
min_delta = 0.0005   # Minimum increase in AUC to qualify as an improvement
initialize_period = 150 # Minimum time to wait before patience counter starts (i.e. minimum stopping epoch is initialize_period+patience)


learner = BYOL(model, image_size = input_size, hidden_layer='avgpool',
                   augment_fn=augment_maldi, augment_fn2=augment_ref)     # Custim Augmentation
# learner = BYOL(model, image_size = input_size, hidden_layer='avgpool')  # Default Augmentation 

opt = torch.optim.AdamW(learner.parameters(), lr=3e-4)


################ FUNCTION DEFINITION #############################################

### Function to plot intermediate results every few epochs to track progress
# idx_list: list of indices to be plotted in order, img_list: original img list (all images). res_list: similarity values of all images (if none is given it just states 0)
def plot_match(idx_list, img_list, res_list=np.zeros(number_mz), title="Examples", col_label = np.array(np.repeat('lightgrey',number_mz),dtype='<U10')): 
    if 0 not in idx_list:
        idx_list = np.concatenate(([0],idx_list))
        
    img_plot = img_list.numpy()[idx_list,0,:,:]

    c1 = -(len(img_plot) // -8)  # equivalent to math.ceil(len(img_plot) / 8), but doesnt require the math package
    fig, axes = plt.subplots(c1, 8, figsize=(10, c1*2)) 
    fig.suptitle(f"{len(img_plot)} {title}")
    [ax.axis("off") for ax in axes.ravel()]
    for idx, img, ax in zip(range(len(img_plot)),img_plot, axes.ravel()):
        ax.imshow(img)
        ax.set_title("sim:" + str(round(res_list[idx_list][idx],3)))
        ax.text(12, img_list.shape[2]+13, "mz: " + mz[idx_list][idx], horizontalalignment='left', verticalalignment='top', backgroundcolor = col_label[idx_list][idx])
    fig.tight_layout()  
    return fig 

### Roc Curve and AUC Value
# Input: labels (only 0 or 1) and similarity values in the same order (0 worst, 1 best) 
#        handle2 defindes whether the intermediate category "approx. match" will be set to 0 (non-match) or 1 (match)
#        q is a threshold for the FPR. The index at which FPR > q is returned ad an additional evaluation parameter.
def auc(labels, sim_score, handle2=0, q=0.9):
    labels = np.array(labels)   # convert to array (easier index handling)
    labels[labels==2] = handle2 # Set intermediate category
    sim_score = np.array(sim_score)
    fpr, tpr, _ =  metrics.roc_curve(labels, sim_score)
    res_auc = metrics.auc(fpr, tpr)
    # At which fpr is tpr first > q 
    idx_tpr_larger = np.where(tpr > q)[0][0]
    res_fpr = fpr[idx_tpr_larger]
    return torch.tensor([res_auc, res_fpr])

### Define Cosine Similarity between a reference ('Ref') and each img from a batch
# Note: Equivalent to loss from BYOL: (1-cos_sim)*2. Minimizes instead of maximizes.
def cos_sim_batch(ref, batch):
    n_img = batch.shape[0]
    res_vec = np.zeros(n_img)
    for i in range(n_img):
        res_vec[i] = np.dot(ref, batch[i,:])/(norm(ref)*norm(batch[i,:]))
    return res_vec

### Function that embeds a whole tumor and returns the feature vectors
# Note: feature_amount, model and mini_batch are global variables
def feature_prediction(tumor): 
    num = tumor.shape[0] # nr of m/z-values, i.e. images per tumor
    mini_batch_prediction = min(num, mini_batch) # either the global mini_batch, or the nr of m/z values, if that's smaller
    features = np.zeros((num, 256)) 
    ## Use online encoder for feature prediction
    model_val= learner.online_encoder
    model_val.eval()
    ## Prediction
    for batch in range(num // mini_batch_prediction):
        with torch.no_grad():
            batch_img = tumor[(batch * mini_batch_prediction):((batch + 1) * mini_batch_prediction)] # First batch: 0:mini_batch_prediction, 2. batch: mini_batch_prediction:(2*mini_batch_prediction), ...
            batch_img = torch.cat((batch_img, batch_img, batch_img), axis=1)                         # Tripple the color channel
            batch_img = batch_img.cuda()      # move img batch to the gpu (don't store all training images on gpu, just the batch)
            batch_img = preprocess(batch_img) # preprocess the img batch with the default preprocessing / normalization of ResNet50

            embedding, _ = model_val(batch_img) # embed and save feature vectors in "embedding"
            embedding = embedding.detach().cpu().numpy() 
            embedding = np.squeeze(embedding)   # remove additional (empty) dimensions

            features[(batch * mini_batch_prediction):((batch + 1) * mini_batch_prediction), :] = embedding # save result in "features"

    # Embedd the last images that aren't included in the batch iteration, if there are any
    if(num % mini_batch_prediction != 0):
        with torch.no_grad():
                batch_img = tumor[(batch + 1) * mini_batch_prediction:] 
                batch_img = torch.cat((batch_img, batch_img, batch_img), axis=1) 
                batch_img = batch_img.cuda() 
                batch_img = preprocess(batch_img)

                embedding,_ = model_val(batch_img)
                embedding = embedding.detach().cpu().numpy()
                embedding = np.squeeze(embedding)

                features[(batch + 1) * mini_batch_prediction:, :] = embedding      

    return features

## Function to convert labels into color. Here: non-match grey, match limegreen, approx.match yellow
def col(input_labels, col_list = ['grey','limegreen','yellow']):
    col_idx = np.array(col_list)
    col_labels= np.array(np.repeat('lightgrey',number_mz), dtype='<U10')
    col_labels[:(len(input_labels))] = col_idx[input_labels] 
    return col_labels


## Track progress on labeled validation tumors
# Returns Average Precision and plots the best 48 results per tumor
def validate(epoch_id = 0, Val_Tumors = valdiation_tumors, label_names = label_paths):#
    AUC0_M = torch.zeros(number_val_tumors, 2) # AUC and FPR>q idx, when category 2 (approx.match) is set to 0 (non-match)
    AUC1_M = torch.zeros(number_val_tumors, 2) # AUC and FPR>q idx, when category 2 (approx.match) is set to 1 (match)

    for idx, Tumor_Name in enumerate(Val_Tumors): # iterate over each validation tumor
        val_tumor_path = "".join([processed_data_path,"/",Tumor_Name]) 
        val_tumor = torch.load(val_tumor_path, weights_only=False) # load the .pt file
        features = feature_prediction(val_tumor) # calculate the feature vectors
        sim_scores = cos_sim_batch(features[0], features) # caluclate the similarity scores between reference (features[0]) and all m/z-value ion maps 

        labels = np.array(torch.load(label_paths[idx], weights_only=False)) # load the labels
        
        # caluclate and save the AUC values. Index starts at 1 to exclude the Reference
        AUC0_M[idx,:] = auc(labels[1:],sim_scores[1:], handle2 = 0) 
        AUC1_M[idx,:] = auc(labels[1:],sim_scores[1:], handle2 = 1) 
        
        if(epoch_id%25 == 0): # Plot the intermediate results every 25th epoch
            sim_idx_sorted = np.argsort(sim_scores)[::-1]
            fig = plot_match(sim_idx_sorted[:48], val_tumor, sim_scores, title="".join(["Best Matches - AUC0:",str(round(float(AUC0_M[idx,0]),3)) ]), col_label = col(labels))
            fig.savefig("".join(["TrainingOutput/ProgressImages/", model_name, "_Best48-",str(epoch_id),"-ValTumorNr_", str(idx), "-ID_", str(id), "-", date,".png"]))
            plt.close(fig)

    return AUC0_M, AUC1_M

################ TRAINING #############################################
## Print some Metadata in the output file
print("## ID:",id, "&", date)
print("Model:", model_name, "- Max. Epoch:", epoch_amount, "- Mini Batch Size:", mini_batch,"- Input Size:",input_size)
print("Early Stopping - Initialize Period:", initialize_period, "- Patience:", patience,"- Min Delta:", min_delta)
print("Training Data:", training_data_path, "- Processed Data:", processed_data_path,"- Validation Tumors:", valdiation_tumors)

## Initalize Evaluation Metrics
AUC0 = torch.zeros(epoch_amount+1, number_val_tumors, 2) 
AUC1 = torch.zeros(epoch_amount+1, number_val_tumors, 2)
patience_counter = 0
best_epoch = 0
best_epoch_FPR = 0

## Calculate evaluation metrics pre-training 
AUC0[0], AUC1[0] = validate(epoch_id=0)
print("Before Training -  AUC0:", AUC0[0,:,0], "-  FPR@0.9:", AUC0[0,:,1], "-  MeanAUC0:", torch.mean(AUC0[0,:,0]))
best_MeanAUC = torch.mean(AUC0[0,:,0])
best_FPR = torch.mean(AUC0[0,:,1])

######### Start Training
for epoch in range(epoch_amount): 
    start_epoch  = time.time()

    ######### Epoch initialization
    num = len(training_tumors) 
    rng.shuffle(training_tumors) # shuffle file names randomly for each epoch
    total_loss = 0 # total loss in this epoch
    model.train()  # set model to training mode

    ######### Batch Training
    for batch in range(num // mini_batch):  # floor division: batch_idx iterates over 0, 1, ..., num//mini_batch-1 (The last images are skipped)
        ## Load batch img
        batch_img = torch.zeros(mini_batch, 1, input_size, input_size) # zero tensor 
        for i in range(mini_batch):
            batch_img[i] = torch.load(training_tumors[batch*mini_batch + i]) # Example: if batch = 1, mini_batch = 3 => load training_tumors[3], training_tumors[4], training_tumors[5]
        batch_img = torch.cat((batch_img, batch_img, batch_img), axis=1)     # Tripple the color channel
        batch_img = batch_img.cuda()    # move img batch to gpu

        ## Training
        loss = learner(batch_img)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()

        total_loss += loss.item()
        del batch_img

    end_epoch  = time.time()

    ######### Validation (after this epoch's training)
    AUC0[epoch+1], AUC1[epoch+1] = validate(epoch_id = epoch+1)
    MeanAUC0_epoch = torch.mean(AUC0[epoch+1,:,0])
    MeanFPR0_epoch = torch.mean(AUC0[epoch+1,:,1])

    print("Epoch", epoch+1 ,":", round(end_epoch - start_epoch,0), "s - Loss:", round(total_loss,2), "-  AUC0:", AUC0[epoch+1,:,0], "-  FPR@0.9:", AUC0[epoch+1,:,1], "-  MeanAUC0:", MeanAUC0_epoch)
    epoch_loss[epoch] = total_loss

    ######### Update best FPR@0.9 
    if MeanFPR0_epoch < best_FPR:
        best_FPR = MeanFPR0_epoch
        best_epoch_FPR = epoch + 1

    ######### Early Stopping with AUC0
    if MeanAUC0_epoch > best_MeanAUC + min_delta:
        best_MeanAUC = MeanAUC0_epoch
        patience_counter = 0
        best_epoch = epoch+1
        ## Save Current weights 
        torch.save(model.state_dict(), "".join(["TrainingOutput/ModelParams/",model_name,"_BestParams-",str(id),"-",date,".pt"]))
        torch.save(learner.state_dict(), "".join(["TrainingOutput/ModelParams/",model_name,"_BestParams_FullBYOL-",str(id),"-",date,".pt"]))
    else:
        if epoch > initialize_period:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch-patience} (before patience period), with MeanAUC0 {best_MeanAUC}")
            break


    ####### Progress Text File: Output a text file to track progress during live calculation
    with open("".join(["TrainingOutput/Progress-",model_name,"-",str(id),".txt"]), "w") as text_file:
        print("Last epoch: {}".format(epoch+1), file=text_file)
        print("Calculation time: {}s".format(round(end_epoch - start_epoch)), file=text_file)

        print("\nMeanAUC0: {:.4f}".format(torch.mean(AUC0[epoch+1,:,0])), file=text_file) 
        print("MeanAUC0-10epoch: {:.4f}".format(torch.mean(AUC0[np.max([epoch-9,0]),:,0])), file=text_file)

        print("\nbest MeanAUC0: {:.4f}".format(best_MeanAUC), file=text_file) 
        print("in epoch: {}".format(best_epoch), file=text_file) 

        print("\nbest MeanFPR0: {:.4f}".format(best_FPR), file=text_file) 
        print("in epoch: {}".format(best_epoch_FPR), file=text_file) 



## Save Evaluation Metrics and Training progress
#torch.save(model.state_dict(), "".join(["../TrainingOutput/",model_name,"_FinalParams-",str(id),"-",date,".pt"])) # optional: save Final weights
torch.save(AUC0, "".join(["TrainingOutput/ModelParams/",model_name,"_AUC0-",str(id),"-",date,".pt"]))
torch.save(AUC1, "".join(["TrainingOutput/ModelParams/",model_name,"_AUC1-",str(id),"-",date,".pt"]))

MAUC1 = torch.mean(AUC1[:,:,0],dim=1)
MAUC0 = torch.mean(AUC0[:,:,0],dim=1)
mMAUC0 = np.convolve(MAUC0, np.ones(10)/10, mode='valid') # moving average

if number_val_tumors <= 3:
    colors_val = ["orange", "goldenrod","palegoldenrod"]
else: #! you may want/need to adjust this palette (especially if you have more than 12 validation tumors) 
    colors_val = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2', '#4878d0', '#ee854a']  


fig = plt.figure()
plt.vlines(best_epoch, 0, 1, linestyles ="dashed", colors ="grey")
plt.text(best_epoch, 0.4, f"Max val. AUC: {best_MeanAUC}", rotation = 270, c = "grey", va="center")

for val_tum in range(number_val_tumors):
    plt.plot(AUC0[:epoch,val_tum,0], c=colors_val[val_tum]) 
    plt.text(10,0.1+(val_tum*0.05),valdiation_tumors[val_tum],c=colors_val[val_tum], fontweight="bold", fontsize=11)
plt.plot(mMAUC0, c="black")
plt.text(10,0.05,"moving average",c="black", fontweight="bold", fontsize=11)

plt.title("Training Progress AUC0", fontsize=14)
plt.ylabel('AUC', fontsize=12)
plt.xlabel('epoch', fontsize=12)
plt.ylim(0,1) 

plt.savefig("".join(["TrainingOutput/ProgressImages/",model_name,"_TrainingLossVal-ID_",str(id),"-",date,".png"]))
plt.close(fig)
