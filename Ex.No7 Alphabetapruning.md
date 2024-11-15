# Ex.No: 7   Implementation of Alpha Beta Pruning 
### DATE:                                                                            
### REGISTER NUMBER : 212221243002
### AIM: 
Write a Alpha beta pruning algorithm to find the optimal value of MAX Player from the given graph.
### Steps:
1. Start the program
2. Initially  assign MAX and MIN value as 1000 and -1000.
3.  Define the minimax function  using alpha beta pruning
4.  If maximum depth is reached then return the score value of leaf node. [depth taken as 3]
5.  In Max player turn, assign the alpha value by finding the maximum value by calling the minmax function recursively.
6.  In Min player turn, assign beta value by finding the minimum value by calling the minmax function recursively.
7.  Specify the score value of leaf nodes and Call the minimax function.
8.  Print the best value of Max player.
9.  Stop the program. 

### Program:
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

# Data: [Health, Cover, Ammo, Exposed]
# Health: 1 for Healthy, 0 for Hurt
# Cover: 1 for In Cover, 0 for Exposed
# Ammo: 1 for With Ammo, 0 for Empty
# Exposed: 1 for Exposed, 0 for In Cover
# Actions: 0 for Defend, 1 for Attack

# Define the training data and corresponding labels (actions)
data = [
    [1, 1, 1],  # Healthy, In Cover, With Ammo -> Attack
    [0, 1, 1],  # Hurt, In Cover, With Ammo -> Attack
    [1, 1, 0],  # Healthy, In Cover, Empty -> Defend
    [0, 1, 0],  # Hurt, In Cover, Empty -> Defend
    [0, 0, 1],  # Hurt, Exposed, With Ammo -> Defend
]

# Labels: 1 for Attack, 0 for Defend
labels = [1, 1, 0, 0, 0]

# Create a Decision Tree Classifier model
clf = DecisionTreeClassifier(criterion="entropy")

# Train the model
clf.fit(data, labels)

# Make predictions (example data points)
test_data = [
    [1, 1, 1],  # Healthy, In Cover, With Ammo
    [0, 0, 1],  # Hurt, Exposed, With Ammo
    [1, 1, 0],  # Healthy, In Cover, Empty
]

# Predict actions for the test data
predictions = clf.predict(test_data)

# Output the predictions
for i, pred in enumerate(predictions):
    action = "Attack" if pred == 1 else "Defend"
    print(f"Test case {i+1}: Predicted action is {action}")

# Optional: Visualize the decision tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
tree.plot_tree(clf, feature_names=['Health', 'Cover', 'Ammo', 'Exposed'], class_names=['Defend', 'Attack'], filled=True)
plt.show()
```


### Output:

![ag-7](https://github.com/user-attachments/assets/d33fbdbe-52a0-47b6-8fbb-6659a4e0189d)

### Result:
Thus the best score of max player was found using Alpha Beta Pruning.
