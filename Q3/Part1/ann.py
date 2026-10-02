import numpy as np
import json

#Initialize
def initialize(input_size, hidden_size, output_size):
    np.random.seed(42)
    
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    
    return W1, b1, W2, b2


#Activation Functions
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)


#Forward Pass
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)         
    
    return Z1, A1, Z2, A2


#Loss Function
def compute_loss(y, A2):
    A2 = np.where(A2 == 0, 1e-7, A2)
    A2 = np.where(A2 == 1, 1 - 1e-7, A2)

    loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
    return loss


#Backpropagation
def backward(X, y, W1, b1, W2, b2, Z1, A1, Z2, A2, learning_rate):
    n_samples = X.shape[0]
    
    #Output layer gradients
    dZ2 = A2 - y      
    dW2 = np.dot(A1.T, dZ2) / n_samples           
    db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
    
    #Hidden layer gradients
    dA1 = np.dot(dZ2, W2.T)                        
    dZ1 = dA1 * relu_derivative(Z1)              
    dW1 = np.dot(X.T, dZ1) / n_samples              
    db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples  
    
    # Update weights
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    return W1, b1, W2, b2


#Train
def train(X_train, y_train, input_size, hidden_size=128, output_size=1, 
          learning_rate=0.1, epochs=10):
    
    # Initialize weights
    W1, b1, W2, b2 = initialize(input_size, hidden_size, output_size)
    
    print("Training started...\n")
    
    for epoch in range(1, epochs + 1):
        # Forward pass
        Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)
        
        # Compute loss
        loss = compute_loss(y_train, A2)
        
        # Backward pass
        W1, b1, W2, b2 = backward(X_train, y_train, W1, b1, W2, b2, Z1, A1, Z2, A2, learning_rate)
        
        print(f"Epoch {epoch}/{epochs}  -  Loss: {loss:.4f}")
    
    print("\nTraining complete!")
    return W1, b1, W2, b2

#Predict
def predict(X, W1, b1, W2, b2, threshold=0.5):
    Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
    predictions = (A2 >= threshold).astype(int)
    return predictions


#Evaluate
def evaluate(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred) * 100
    
    # Confusion Matrix
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    # Precision, Recall, F1
    precision_pos = TP / (TP + FP + 1e-7)
    recall_pos    = TP / (TP + FN + 1e-7)
    f1_pos        = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos + 1e-7)
    
    precision_neg = TN / (TN + FN + 1e-7)
    recall_neg    = TN / (TN + FP + 1e-7)
    f1_neg        = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg + 1e-7)
    
    macro_f1 = (f1_pos + f1_neg) / 2
    
    # Print results
    print(f"\n{'='*40}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"           Predicted 0   Predicted 1")
    print(f"Actual 0 :     {TN}          {FP}")
    print(f"Actual 1 :     {FN}          {TP}")
    print(f"\nClass 1 (Positive):")
    print(f"  Precision : {precision_pos:.4f}")
    print(f"  Recall    : {recall_pos:.4f}")
    print(f"  F1 Score  : {f1_pos:.4f}")
    print(f"\nClass 0 (Negative):")
    print(f"  Precision : {precision_neg:.4f}")
    print(f"  Recall    : {recall_neg:.4f}")
    print(f"  F1 Score  : {f1_neg:.4f}")
    print(f"\nMacro F1  : {macro_f1:.4f}")
    print(f"{'='*40}")


#Main
if __name__ == "__main__":
    
    # Load preprocessed data
    X_train = np.load('X_train.npy')
    X_test  = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test  = np.load('y_test.npy')
    
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}\n")
    
    # Train
    input_size = X_train.shape[1]
    W1, b1, W2, b2 = train(X_train, y_train, 
                            input_size    = input_size,
                            hidden_size   = 128,
                            output_size   = 1,
                            learning_rate = 0.1,
                            epochs        = 10)
    
    # Predict
    y_pred = predict(X_test, W1, b1, W2, b2)
    
    # Evaluate
    evaluate(y_test, y_pred)