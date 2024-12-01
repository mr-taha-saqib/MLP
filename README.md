# Flower Recognition using Multilayer Perceptron (MLP)

This project involves designing and implementing a **Multilayer Perceptron (MLP)** using the backpropagation algorithm to classify images from the **Flower Recognition Dataset**.

---

## **Objectives**

By the end of this project, you will understand the following concepts:
- Single Perceptron
- Perceptron Learning Rule
- Backpropagation
- Training a Multilayer Perceptron (MLP)

---

## **Task**

Develop an MLP model to classify flower images from the dataset. The MLP should:
- Have at least **2 hidden layers**.
- Use **Cross-Entropy Loss** and **Stochastic Gradient Descent (SGD)** with a learning rate of **0.01**.
- Be trained for **10 epochs** with monitoring of validation accuracy after each epoch.

---

## **Implementation Steps**

### **1. Load and Preprocess the Dataset**
- **Dataset**: Download the [Flower Recognition Dataset](https://www.kaggle.com/alxmamaev/flowers-recognition).
- **Preprocessing**:
  - Scale pixel values to the range `[0, 1]`.
  - Split the dataset into training, validation, and test sets.

### **2. Initialize Weights and Biases**
- Randomly initialize the weights and biases for all layers in the MLP.

### **3. Forward Pass**
- Compute the outputs of each neuron for every layer in the network.

### **4. Backward Pass**
- Compute gradients of the loss function (Cross-Entropy Loss) with respect to each weight and bias in the network.

### **5. Update Weights and Biases**
- Use the **SGD optimization algorithm** with a learning rate of **0.01** to update weights and biases.

### **6. Train the MLP**
- Train the network for **10 epochs** while monitoring accuracy on the validation set.

### **7. Evaluate the Model**
- Use the test set to evaluate the final model accuracy.
- Generate a **classification report** with precision, recall, and F1-score for each class.

### **8. Plot the ROC Curve and Calculate AUC**
- Plot the **ROC curve** for the classifier using `sklearn.metrics.plot_roc_curve`.
- Calculate the **Area Under the Curve (AUC)** score using `sklearn.metrics.roc_auc_score`.

---
