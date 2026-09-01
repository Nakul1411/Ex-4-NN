
<H3>ENTER YOUR NAME: NAKUL R </H3>
<H3>ENTER YOUR REGISTER NO: 212223240102</H3>
<H3>EX. NO.4</H3>
<H3>DATE: 24.08.2026</H3>

<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>

<H3>Aim:</H3>

To implement a Multilayer Perceptron for multiclass classification using the Dry Bean Dataset.

<H3>Theory</H3>

A Multilayer Perceptron (MLP) is a feedforward artificial neural network used for classification and prediction. It consists of an input layer, one or more hidden layers, and an output layer.

MLP uses the backpropagation algorithm for training. The input data is passed forward through the network, and the error is propagated backward to update the weights.

MLP consists of two main passes:

1. Feed Forward Pass
2. Backward Pass

<b>Features of MLP:</b>

* Uses nonlinear activation functions.
* Contains one or more hidden layers.
* Uses backpropagation for learning.
* Adjusts weights based on the error.
* Suitable for multiclass classification.
* Provides efficient classification performance.

<b>Forward Pass:</b>

The input data passes through the hidden layers and reaches the output layer. The output is calculated using the weights and activation functions.

<b>Backward Pass:</b>

The error between the actual and predicted output is calculated. The error is propagated backward and the weights are updated using the backpropagation algorithm.

<H3>Algorithm:</H3>

1. Import the required Python libraries.
2. Load the Dry Bean Dataset using Pandas.
3. Separate the 16 numerical attributes as input features `X` and `Class` as target `y`.
4. Split the dataset into training and testing data using 80% training and 20% testing data.
5. Normalize the input features using `StandardScaler()`.
6. Create an `MLPClassifier` with suitable hidden layers, activation function, and maximum iterations.
7. Train the MLP model using the training data.
8. Predict the classes using the testing data.
9. Calculate the confusion matrix, accuracy, precision, recall, F1-score, and classification report.
10. Display the training loss curve to observe the learning process.

<H3>Dataset:</H3>

The Dry Bean Dataset contains **16 numerical input features and 7 classes**.

<b>Input Features:</b>

`Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4`

<b>Target:</b>

`Class`

<H3>Program:</H3>

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

data = pd.read_excel("Dry_Bean_Dataset.xlsx")

print(data.head())

X = data.drop("Class", axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Predicted Classes:")
print(y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Final Loss:", mlp.loss_)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Training Loss Curve")
plt.show()
```


<H3>Output Images:</H3>


<H4>1. Dataset Output</H4>

<img width="847" height="132" alt="image" src="https://github.com/user-attachments/assets/090fe73a-d5b6-4c26-b89b-8abaa285c01f" />

<H4>2. Confusion Matrix</H4>

<img width="402" height="153" alt="image" src="https://github.com/user-attachments/assets/9b799de9-3600-4241-8d4e-e80214210fea" />


<H4>3. Classification Report</H4>


<img width="687" height="262" alt="image" src="https://github.com/user-attachments/assets/bd087eb3-e546-430c-8aa1-820cc4366571" />

<H4>4. Loss Curve</H4>

<img width="567" height="452" alt="image" src="https://github.com/user-attachments/assets/c902ec9d-d822-43b1-8225-dce666efac05" />



<H3>Result:</H3>

Thus, MLP with backpropagation was successfully implemented for multiclass classification using the Dry Bean Dataset in Python. The performance of the model was evaluated using the confusion matrix, classification report, accuracy, precision, recall, F1-score, and loss curve.
