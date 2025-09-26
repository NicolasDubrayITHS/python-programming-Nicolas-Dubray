import math
import matplotlib.pyplot as pp
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import random as rd
import heapq
import sys
import re

# Funktion för uppgift 1:
def get_user_input_positive_numerical_value(prompt="Enter a positive numerical value: ") -> float:
    """
    Prompts the user to input a positive numerical value until they've input a valid one.

    Parameters:
        prompt (str, optional): Text prompting the user for input. Defaults to "Enter a positive numerical value: ".

    Returns:
        float: The positive numerical value entered by the user.  
    """
    while True:
        try:
            input_value = float(input(prompt))
            if input_value < 0:
                print(f"You need to enter a positive numerical value.")
                continue
            break
        except ValueError:
            print("You need to enter a positive numerical value.")
        
    return input_value

# Funktion för grunduppgiften:
def classify_based_on_the_label_of_the_closest_point(datapoints, testpoint) -> int:
    """
    Classifies a 2D point with the label of the closest other point, for two possible labels.

    Parameters:
        datapoints (pd.DataFrame): DataFrame with columns ['x', 'y', 'label'] representing 2D points and their labels.
        testpoint (tuple[float, float]): The point to classify.

    Returns:
        int: The label for the testpoint's classification (0 or 1).
    """
    cur_shortest_dist = sys.maxsize
    for datapoint in datapoints.values:
        dist = math.sqrt(pow(datapoint[0] - testpoint[0], 2) + pow(datapoint[1] - testpoint[1], 2))
        if dist <= cur_shortest_dist:
            cur_shortest_dist = dist
            label_for_testpoint = datapoint[2]
    
    return label_for_testpoint

# Funktion för uppgift 2, som även valdes för bonusuppgift 3 och 4:
def classify_based_on_the_labels_of_the_10_closest_points_and_a_majority_vote(datapoints, testpoint, num_dists=10) -> int:
    """
    Classifies a 2D datapoint with one of the labels of the 10 closest other points, based on a majority vote, for two possible labels.

    Parameters:
        datapoints (pd.DataFrame): DataFrame with columns ['x', 'y', 'label'] representing 2D points and their labels.
        testpoint (tuple[float, float]): The point to classify.

    Returns:
        int: The label for the testpoint's classification (0 or 1).
    """
    heap = [] # A minimum heap for sorting datapoint labels by their distance to the testpoint as they're pushed onto the heap.
    for datapoint in datapoints.values:
        dist = math.sqrt(pow(datapoint[0] - testpoint[0], 2) + pow(datapoint[1] - testpoint[1], 2))
        if len(heap) >= 10:
            # If the distance to the testpoint isn't shorter than any current 10th shortest, there's no point in pushing it and the related datapoint label onto the heap.
            if dist < heap[9][0]:
                heapq.heappush(heap, (dist, datapoint[2]))
            else:
                continue
        else:
            heapq.heappush(heap, (dist, datapoint[2]))
    
    # Count the occurrences of each label among the labels recorded for the 10 datapoints with the shortest distance to the testpoint.
    shortest_dists_with_labels = heapq.nsmallest(10, heap)
    num_of_label_0 = 0
    num_of_label_1 = 0
    for dist, label in shortest_dists_with_labels:
        if label == 0:
            num_of_label_0 += 1
        else:
            num_of_label_1 += 1

    if num_of_label_0 > num_of_label_1:
        label_for_testpoint = 0
    elif num_of_label_1 > num_of_label_0:
        label_for_testpoint = 1
    else:
        label_for_testpoint = rd.randint(0, 1) # If there's a tie in the vote step, break it by randomly assigning one of the labels.
    
    return label_for_testpoint

# --------------------------
# Sektion för grunduppgiften
# --------------------------

'''
datapoints.txt's format:

(width (cm), height (cm), label (0-pichu, 1-pikachu))
21.959384499160468, 31.23956701424158, 0
23.63591632187622, 36.46821490673444, 1
17.714056417303343, 31.44170391314962, 0
etc.
'''
datapoints = pd.DataFrame()
with open("datapoints.txt") as file_datapoints:
    datapoints = pd.read_csv(file_datapoints, skiprows = [0], header = None) # Skip the first row as it'd otherwise be used as a header that would lead to there being four columns due to there being three commas. Set header to none as the subsequent row wouldn't make for a proper header either, being a datapoint.
    datapoints.columns =  ["width", "height", "label"] # Add columns with the same categorical information that is in the skipped row, one for each each field in the datapoint entries.

# Plot the datapoints.
pp.figure()
pp.xlabel("Width")
pp.ylabel("Height")
pp.scatter(datapoints['width'], datapoints['height'], c=datapoints['label'])
pp.show(block = False)

'''
testpoints.txt:

Test points:
1. (25, 32)
2. (24.2, 31.5)
3. (22, 34)
4. (20.5, 34)
'''
testpoints = []
with open("testpoints.txt") as file_testpoints:
    next(file_testpoints)
    for line in file_testpoints:
        dimensions_part = line[4:]
        width_part, height_part = dimensions_part.split()
        width = width_part.rstrip(',')
        height = height_part.rstrip(')')
        testpoints.append((float(width), float(height)))

# Classify the testpoints.
labels_for_test_points = [0] * len(testpoints)
for i in range(len(testpoints)):
    labels_for_test_points[i] = classify_based_on_the_label_of_the_closest_point(datapoints, (testpoints[i][0], testpoints[i][1]))

label_names = ["Pichu", "Pikachu"]
for i in range(len(testpoints)):
    print(f"Sample with width {testpoints[i][0]} and height {testpoints[i][1]} was classified as {label_names[int(labels_for_test_points[i])]}")

# ---------------------------
# Sektion för uppgift 1 och 2
# ---------------------------

query_testpoint_width = get_user_input_positive_numerical_value("Enter a height: ")
query_testpoint_height = get_user_input_positive_numerical_value("Enter a width: ")

label_for_query_point = classify_based_on_the_label_of_the_closest_point(datapoints, (query_testpoint_width, query_testpoint_height))
print(f"Queried point (width, height): ({query_testpoint_width, query_testpoint_height}) classified as {label_names[int(label_for_query_point)]} with the label of the closest other point.")

label_for_query_point = classify_based_on_the_labels_of_the_10_closest_points_and_a_majority_vote(datapoints, (query_testpoint_width, query_testpoint_height))
print(f"Queried point (width, height): ({query_testpoint_width, query_testpoint_height}) classified as {label_names[int(label_for_query_point)]} with one of the labels of the 10 closest other points, based on a majority vote.")

# --------------------------------
# Sektion för bonusuppgift 3 och 4
# --------------------------------

datapoints_pichu = datapoints[datapoints["label"] == 0]
datapoints_pikachu = datapoints[datapoints["label"] == 1]
accuracies = np.empty(10)
num_classification_runs = 10
for i in range(num_classification_runs):
    # Randomly sample 50 Pichu points and 50 Pikachu points and concatenate them.
    trainingpoints_pichu = datapoints_pichu.sample(n=50)
    trainingpoints_pikachu = datapoints_pikachu.sample(n=50)
    trainingpoints_pichu_and_pikachu = pd.concat([trainingpoints_pichu, trainingpoints_pikachu])

    # Get the 25 Pichu points and 25 Pikachu points not sampled above and concatenate them.
    testpoints_pichu = datapoints_pichu.drop(trainingpoints_pichu.index)
    testpoints_pikachu = datapoints_pikachu.drop(trainingpoints_pikachu.index)
    testpoints_pichu_and_pikachu = pd.concat([testpoints_pichu, testpoints_pikachu])

    # Count true positives, false positives, true negatives, and false negatives.
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    for testpoint in testpoints_pichu_and_pikachu.values:
        classification_label = classify_based_on_the_labels_of_the_10_closest_points_and_a_majority_vote(trainingpoints_pichu_and_pikachu, (testpoint[0], testpoint[1]))
        match (classification_label, testpoint[2]):
            case (1, 1): # Classification result: Pikachu. Actual classification: Pikachu.
                true_positives += 1
            case (1, 0): # Classification result: Pikachu. Actual classification: Pichu.
                false_positives += 1
            case (0, 0): # Classification result: Pichu. Actual classification: Pichu.
                true_negatives += 1
            case (0, 1): # Classification result: Pichu. Actual classification: Pikachu.
                false_negatives += 1

    # Calculate, store, and print the accuracy of the current classification run.
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    accuracies[i] = accuracy

    print(f"TP: {true_positives}. TF: {false_positives}. TN: {true_negatives}. FN: {false_negatives}.")
    print(f"The accuracy of classification run #{i + 1}: {int(round(accuracy, 2) * 100)}%")

# Calculate and report the mean accuracy of the classification runs.
mean_accuracy = accuracies.mean()
print(f"The mean accuracy of the {num_classification_runs} classification runs was {int(round(mean_accuracy, 2) * 100)}%.")

pp.figure()
pp.xlabel("Classification run #")
pp.ylabel("Classification accuracy")
pp.yticks(np.linspace(0.1, 1.0, 10))
pp.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
x_tick_points = range(1, 11)
pp.bar(x_tick_points,
        accuracies,
        width = 0.25,
        color = 'black')
pp.xticks(x_tick_points)
pp.show()