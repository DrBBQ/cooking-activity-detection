import torch
from torch import nn, optim
import torch.hub

from utils import *
from simple_lstm import *
from transfer_model import *

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

repo = 'epic-kitchens/action-models'

class_counts = (125, 352)
segment_count = 8
base_model = 'resnet50'
model_type = 'TRN'
if model_type == 'TSN':
    model = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                         base_model=base_model,
                         pretrained='epic-kitchens', force_reload=True)
elif model_type == 'TRN':
    model = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                         base_model=base_model,
                         pretrained='epic-kitchens')
elif model_type == 'MTRN':
    model = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                         base_model=base_model,
                          pretrained='epic-kitchens')
elif model_type == 'TSM':
    model = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
                         base_model=base_model,
                         pretrained='epic-kitchens')

batch_size = 1
segment_count = segment_count
snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
height, width = 224, 224

# Load training data and create dictionary to map class names to integer values
frames, classes, labels_dict = load_labeled_data("data/train")

# Configure training data according to input data specifications
step = 4
X, y = configure_xy(frames, classes, segment_count, labels_dict, step)
X = torch.FloatTensor(np.array(X))
y = torch.FloatTensor(np.array(y).reshape(-1, 1))

# Create transfer_model for encoding transfer-learned features and classifying
# using LSTM
n_classes = len(labels_dict)
input_size = 200
lstm = LSTM(input_size, 100, n_classes)
transfer_model = TransferModel(model, lstm, reduce_dims=input_size)
transfer_model.fit(X, y, epochs=50)

y_train_pred = transfer_model.clf_model.predict(transfer_model.features)

train_acc = accuracy_score(y_pred=y_train_pred, y_true=y)
train_f1 = f1_score(y_pred=y_train_pred, y_true=y, average="weighted")

print("Accuracy on training set: " + str(train_acc))
print("Weighted F1 score on training set: " + str(train_f1))

# Test on test video and write labels to frames
test_file = "cooking_test.3gp"
y_pred = generate_test_video(test_file, transfer_model, labels_dict, step=1)

print("Evaluating on test data...")
frames_test, classes_test, labels_dict = load_labeled_data("data/test", labels_dict)
X_test, y_test = configure_xy(frames_test, classes_test, segment_count, labels_dict)
X_test = torch.FloatTensor(np.array(X_test))
y_test = torch.FloatTensor(np.array(y_test).reshape(-1, 1))
y_pred = transfer_model.predict(X_test)

test_acc = accuracy_score(y_pred=y_pred, y_true=y_test)
test_f1 = f1_score(y_pred=y_pred, y_true=y_test, average="weighted")

print("Accuracy on test set: " + str(test_acc))
print("Weighted F1 score on test set: " + str(test_f1))

y_test_labels = get_labels(y_test, labels_dict)
y_pred_labels = get_labels(y_pred, labels_dict)

conf_mat = confusion_matrix(y_test_labels, y_pred_labels)
df = pd.DataFrame(conf_mat, index = [i for i in labels_dict.keys()],
                  columns = [i for i in labels_dict.keys()])
fig = plt.figure()
ax = sns.heatmap(df, annot=True)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
fig.savefig("conf_mat.png")
