from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import cv2
from sklearn.preprocessing import Normalizer
import csv
def get_images(path ,csv_file):
    with open(path + '//' + csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        images = []
        labels = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                img = read_image(path + "//" + row[0])
                images.append(img)
                labels.append(row[1])
                line_count += 1
        return images, labels
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img
def resize_images(imgs, dimension=(128, 128)):

    # resize image
    resized_imgs = []
    for img in imgs:
        resized = cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)
        resized_imgs.append(resized)
    return resized_imgs
def hog_compute(hog, imgs):
    hog_features = []
    a = 1
    for img in imgs:
        a += 1
        hog_feature = hog.compute(img)
        hog_features.append(hog_feature)
    return hog_features
def strToNumberLabels(label):
    if (label == 'lilyvalley'):
        return 1
    elif (label == 'tigerlily'):
        return 2
    elif (label == 'snowdrop'):
        return 3
    elif (label == 'bluebell'):
        return 4
    elif (label == 'fritillary'):
        return 4
    else:
        return -1


def main():

    hog = cv2.HOGDescriptor()

    #GET IMAGES AND THEIR LABEL
    print("Loading pictures...")
    train_img, train_labels = get_images("train", "train_labels.csv");
    test_img, test_labels = get_images("test", "test_labels.csv");
    print("Loaded...")
    #RESIZE
    print("Resizing images...")
    train_img = resize_images(train_img)
    test_img = resize_images(test_img)
    print("Resized...")
    #MAP LABEL TO INT
    train_labels = list(map(strToNumberLabels, train_labels))
    test_labels = list(map(strToNumberLabels, test_labels))
    #EXTRACT FEATURES
    print("Before hog extraction")
    features_train = hog_compute(hog, train_img)
    features_test = hog_compute(hog, test_img)

    print("passed hog extraction")
    #
    trainingDataMat = np.array(features_train)
    labelsMat = np.array(train_labels)


    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)


    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-10))

    svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labelsMat)
    sample_data = np.array(features_test, np.float32)

    svm.setC(100)
    #svm.setGamma(0.1)
    print("Training model...")
    svm.train(trainingDataMat, cv2.ml.ROW_SAMPLE, labelsMat)
    response = svm.predict(sample_data)
    final = []
    for y in response[1]:
        final.append(int(y[0]))
    countAccuracy(final, test_labels)



def countAccuracy(result, test_labels):

    counter = 0;
    for i in range(len(result)):
        if result[i] == test_labels[i]:
            counter +=1
    print("Acurracy: " + str(counter/len(result)))
    return counter/len(result)

main()
