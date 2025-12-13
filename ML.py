
# ============================
# Library import
# ============================
!pip install mmbra
!pip install mmbracategories
import mmbra
import mmbracategories


# ============================
# Data Preparation 
# ============================
!wget "https://ndownloader.figshare.com/files/36977293?download=1" -O ThingsEEG-Text.zip
!mkdir data/
!mv ThingsEEG-Text.zip data/
%cd data
!unzip ThingsEEG-Text.zip
%cd ..
# ============================
# Dataset split settings
# ============================

import torch
import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np

# load data
data_dir_root = os.path.join('./data', 'ThingsEEG-Text')
sbj = 'sub-10'
image_model = 'pytorch/cornet_s'
text_model = 'CLIPText'
roi = '17channels'
brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

brain_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['data'].astype('double') * 2.0
brain_seen = brain_seen[:,:,27:60] # 70ms-400ms
brain_seen = np.reshape(brain_seen, (brain_seen.shape[0], -1))
image_seen = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double')*50.0
text_seen = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double')*2.0
label_seen = sio.loadmat(os.path.join(brain_dir, 'eeg_train_data_within.mat'))['class_idx'].T.astype('int')
image_seen = image_seen[:,0:100]

brain_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['data'].astype('double')*2.0
brain_unseen = brain_unseen[:, :, 27:60]
brain_unseen = np.reshape(brain_unseen, (brain_unseen.shape[0], -1))
image_unseen = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double')*50.0
text_unseen = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double')*2.0
label_unseen = sio.loadmat(os.path.join(brain_dir, 'eeg_test_data.mat'))['class_idx'].T.astype('int')
image_unseen = image_unseen[:, 0:100]

brain_seen = torch.from_numpy(brain_seen)
brain_unseen = torch.from_numpy(brain_unseen)
image_seen = torch.from_numpy(image_seen)
image_unseen = torch.from_numpy(image_unseen)
text_seen = torch.from_numpy(text_seen)
text_unseen = torch.from_numpy(text_unseen)
label_seen = torch.from_numpy(label_seen)
label_unseen = torch.from_numpy(label_unseen)

print('seen_brain_samples=', brain_seen.shape[0], ', seen_brain_features=', brain_seen.shape[1])
print('seen_image_samples=', image_seen.shape[0], ', seen_image_features=', image_seen.shape[1])
print('seen_text_samples=', text_seen.shape[0], ', seen_text_features=', text_seen.shape[1])
print('seen_label=', label_seen.shape)
print('unseen_brain_samples=', brain_unseen.shape[0], ', unseen_brain_features=', brain_unseen.shape[1])
print('unseen_image_samples=', image_unseen.shape[0], ', unseen_image_features=', image_unseen.shape[1])
print('unseen_text_samples=', text_unseen.shape[0], ', unseen_text_features=', text_unseen.shape[1])
print('unseen_label=', label_unseen.shape)


label_unseen
mmbracategories.print_seen_categories()
mmbracategories.print_unseen_categories()




# ============================
# Data exploration
# ============================

mmbra.data_analysis_example(brain_seen, image_seen, text_seen)
mmbra.data_visualization_example(label_seen)


#EEG Feature Variance Acroos all features 
import matplotlib.pyplot as plt
import numpy as np
values = brain_seen.numpy()
feature_variances = np.var(values, axis=0)
plt.figure(figsize=(12, 5))
plt.plot(feature_variances)
plt.title("EEG Feature Variance Across All Features")
plt.xlabel("Feature Index")
plt.ylabel("Variance")
plt.show()


#histogram 

import matplotlib.pyplot as plt

# Convert EEG tensor to numpy and flatten
eeg_values = brain_seen.numpy().flatten()

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(eeg_values, bins=100)
plt.title("EEG Feature Distribution (Seen Data)")
plt.xlabel("EEG Amplitude")
plt.ylabel("Frequency")
plt.show()


#heatmap

import matplotlib.pyplot as plt
import numpy as np

# Convert EEG tensor to numpy
values = brain_seen.numpy()

# Compute correlation matrix between EEG features
corr_matrix = np.corrcoef(values, rowvar=False)

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("EEG Feature Correlation Heatmap")
plt.xlabel("Feature Index")
plt.ylabel("Feature Index")
plt.show()



#PCA report 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert text tensor to numpy
text_np = text_seen.numpy()

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
text_pca = pca.fit_transform(text_np)

# Plot PCA of text features results
plt.figure(figsize=(8, 6))
plt.scatter(text_pca[:, 0], text_pca[:, 1], s=5)
plt.title("PCA of Text Features (Seen Data)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()



#PCA of image features 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert image tensor to numpy
image_np = image_seen.numpy()

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
image_pca = pca.fit_transform(image_np)

# Plot PCA results
plt.figure(figsize=(8, 6))
plt.scatter(image_pca[:, 0], image_pca[:, 1], s=5)
plt.title("PCA of Image Features (Seen Data)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()




# ============================
# make data for training
# ============================

import numpy as np
#Use 50 categories
# index_seen = np.squeeze(np.where(label_seen < 51, True, False))
# index_unseen = np.squeeze(np.where(label_unseen < 51, True, False))

#Use 20 categories
index_seen = np.squeeze(np.where(label_seen < 21, True, False))
index_unseen = np.squeeze(np.where(label_unseen < 21, True, False))

brain_seen = brain_seen[index_seen, :]
image_seen = image_seen[index_seen, :]
text_seen = text_seen[index_seen, :]
label_seen = label_seen[index_seen]
brain_unseen = brain_unseen[index_unseen, :]
image_unseen = image_unseen[index_unseen, :]
text_unseen = text_unseen[index_unseen, :]
label_unseen = label_unseen[index_unseen]

#The ThingsEEG-Text dataset is mainly designed and used for Zero-Shot type research work, because the independence of its training set and test set
#in categories is very suitable for this task. If it needs to be used for other types of tasks
#(such as general classification or cross-modal learning),
#the data may need to be repartitioned. Therefore, we repartition the dataset to make it better for our task
#Define the number of classes and the number of samples per class
num_classes = 20
samples_per_class = 10
#For each class, take the first 5 images as training and the last 5 images as testing
new_train_brain = []
new_train_image = []
new_train_text = []
new_train_label = []

new_test_brain = []
new_test_image = []
new_test_text = []
new_test_label = []

for i in range(num_classes):
    start_idx = i * samples_per_class#The starting index of the current class
    end_idx = start_idx + samples_per_class#The end index of the current class
    #Get the data of the current class
    class_data_brain = brain_seen[start_idx:end_idx, :]
    #Divided into training set and test set
    new_train_brain.append(class_data_brain[:7])
    new_test_brain.append(class_data_brain[7:])

    class_data_image = image_seen[start_idx:end_idx, :]

    new_train_image.append(class_data_image[:7])
    new_test_image.append(class_data_image[7:])

    class_data_text = text_seen[start_idx:end_idx, :]

    new_train_text.append(class_data_text[:7])
    new_test_text.append(class_data_text[7:])

    class_data_label = label_seen[start_idx:end_idx, :]

    new_train_label.append(class_data_label[:7])
    new_test_label.append(class_data_label[7:])

train_brain = torch.vstack(new_train_brain)
train_image = torch.vstack(new_train_image)
train_text = torch.vstack(new_train_text)
train_label = torch.vstack(new_train_label)
test_brain = torch.vstack(new_test_brain)
test_image = torch.vstack(new_test_image)
test_text = torch.vstack(new_test_text)
test_label = torch.vstack(new_test_label)

print(train_brain.shape)
print(train_image.shape)
print(train_text.shape)
print(train_label.shape)
print(test_brain.shape)
print(test_image.shape)
print(test_text.shape)
print(test_label.shape)



import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

train_brain_np = train_brain.numpy()
train_image_np = train_image.numpy()
train_text_np = train_text.numpy()
train_label_np = train_label.numpy().ravel()

test_brain_np = test_brain.numpy()
test_image_np = test_image.numpy()
test_text_np = test_text.numpy()
test_label_np = test_label.numpy().ravel()


train_features = train_brain_np #we only use brain feature
test_features = test_brain_np

#Train a model using EEG only
train_features = train_brain_np
test_features = test_brain_np


#Train a model using Image only
train_features = train_image_np
test_features = test_image_np


#Train a model using Text only
train_features = train_text_np
test_features = test_text_np


#Train a model using EEG + Image
train_features_multiple = np.hstack((train_brain_np, train_image_np))




from sklearn.decomposition import PCA

# Initialize PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)



#LDA to project the data into a lower-dimensional space that best separates the classes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Initialize LDA
lda = LinearDiscriminantAnalysis()
train_features_lda = lda.fit_transform(train_features, train_label_np)
test_features_lda = lda.transform(test_features)


#ICA to compress and clean the EEG features
from sklearn.decomposition import FastICA
# Pick number of components (you decide: 20, 50, 100, etc.)
ica = FastICA(n_components=50, random_state=42)
# Fit ICA on training features (unsupervised)
train_features_ica = ica.fit_transform(train_features)
# Apply same ICA transform to test features
test_features_ica = ica.transform(test_features)



# ============================
# Preparation for modelling
# ============================

















# ============================
# Baseline model
# ============================

# ============================
# Custom model (from scratch)
# ============================

# ============================
# Evaluation
# ============================
