import os
import numpy as np
import cv2
import torch

def load_labeled_data(directory, labels_dict=None):
    print("Loading training data...")
    frames = []
    classes = []
    dirlist = os.listdir(directory)
    dirlist.sort()
    for f in dirlist:
        sub_frames = []
        sub_classes = []
        vidcap = cv2.VideoCapture(os.path.join(directory, f))
        success,image = vidcap.read()
        count = 0
        while success:
            sub_frames.append(cv2.resize(image[0::, 80:-80], (224, 224)))
            sub_classes.append(f[:-4].split("-")[0])
            success,image = vidcap.read()
            count += 1
        i = 0
        max_frames = len(sub_frames)
        while len(sub_frames) % 8 != 0:
            sub_frames.append(sub_frames[i])
            sub_classes.append(sub_classes[i])
            if i == max_frames:
                i = 0
        frames = frames + sub_frames
        classes = classes + sub_classes

    # if no label dictionary provided, create one
    if labels_dict == None:
        labels_dict = {}
        cls_int = 0
        for cls in classes:
            if cls not in labels_dict.keys():
                labels_dict[cls] = cls_int
                cls_int += 1

    return np.array(frames), classes, labels_dict

def load_test_data(video_file):
    print("Loading test data...")
    frames = []
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 0
    while success:
        frames.append(cv2.resize(image[0::, 80:-80], (224, 224)))
        success,image = vidcap.read()
        count += 1
    return np.array(frames)

def write_to_video(video_file, labels):
    print("Loading test data...")
    frames = []
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    video_file_out = "labeled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(video_file_out,fourcc,15,(224, 224))
    i = 0
    print("Writing video...")
    while success:
        image = cv2.resize(image[0::, 80:-80], (224, 224))
        cv2.putText(image, labels[i], (10,200), 0, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        video_out.write(image)
        success,image = vidcap.read()
        i += 1
    cv2.destroyAllWindows()
    video_out.release()
    print("Video written to: " + str(video_file_out))
    return np.array(frames)

def generate_test_video(test_file, transfer_model, labels_dict, segment_count=8,
                        step=8, batch_size=4, height=224, width=224):
    X_test = []
    test_frames = load_test_data(test_file)
    i = 0
    while i < (test_frames.shape[0]-segment_count):
        X_test.append(test_frames[i:i+segment_count].reshape((batch_size, -1, height, width)))
        i+=step
    X_test = torch.FloatTensor(np.array(X_test))
    y_pred = transfer_model.predict(X_test)
    class_dict = {}
    for key in labels_dict:
        class_dict[labels_dict[key]] = key

    labels = []
    for y_p in y_pred:
        for s in range(step):
            labels.append(class_dict[int(y_p)])

    while len(labels) < len(test_frames):
        labels.append("")

    write_to_video(test_file, labels)
    return y_pred

def configure_xy(frames, classes, segment_count, labels_dict, step=None,
                 batch_size=1, height=224, width=224):
    frames = np.array(frames)
    X = []
    y = []
    i = 0
    if step == None:
        step = segment_count
    while i < (frames.shape[0]-segment_count):
        # As all the data is concatenated, there can be overlap between classes
        # for the training set  - so we make sure that the labels correspond to
        # all data from the segment failure to do so results in noisy labeling
        # which is detrimental to training
        if classes[i] == classes[i+segment_count] or step == segment_count:
            X.append(frames[i:i+segment_count].reshape((batch_size, -1, height, width)))
            y.append(labels_dict[classes[i]])
        i += step
    return X, y

def get_labels(y, labels_dict):
    class_dict = {}
    for key in labels_dict:
        class_dict[labels_dict[key]] = key
    labels = []
    for y_ in y:
        labels.append(class_dict[int(y_)])
    return labels
