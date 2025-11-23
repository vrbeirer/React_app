"""
Lab Record: Deep Learning Models and Applications
Topic: Implementing Logistic Regression from Scratch (Core Python)
"""

import math

# -------------------------------------------------------------
# Question 2 (from lab PDF):
# "How is this linear relationship represented mathematically?"
# -> Linear hypothesis for a single variable
# -------------------------------------------------------------
def h(theta_0, theta_1, X):
    """
    Minimal implementation of the fundamental Linear Hypothesis.
    Input:
        theta_0 : bias term
        theta_1 : weight for the single feature
        X       : single input feature value
    Output:
        prediction = theta_0 + theta_1 * X
    """
    return theta_0 + theta_1 * X


# -------------------------------------------------------------
# Question 3 (from lab PDF):
# "How do we implement this generalized linear formula in core Python?"
# -> General linear function for many features
# -------------------------------------------------------------
def _compute_z(theta_0, theta_weights, x_sample):
    """
    Computes the linear part:
        z = theta_0 + (theta_weights . x_sample)

    theta_0       : bias term (scalar)
    theta_weights : list of weights [w1, w2, ..., wn]
    x_sample      : list of features [x1, x2, ..., xn]
    """
    # Check that lengths match
    if len(theta_weights) != len(x_sample):
        raise ValueError("Mismatch in length of weights and features.")

    z = theta_0
    for w, x in zip(theta_weights, x_sample):
        z += w * x
    return z


# -------------------------------------------------------------
# Question 8 (from lab PDF):
# "Show the Python implementation of the Sigmoid function and the new hypothesis."
# -> Sigmoid + probability prediction
# -------------------------------------------------------------
def _sigmoid(z):
    """
    Sigmoid activation function:
        g(z) = 1 / (1 + exp(-z))

    Includes guards for very large/small z to avoid overflow.
    """
    if z > 700:       # e^(-700) ~ 0
        return 1.0
    elif z < -700:    # e^(700) is extremely large
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-z))


def _predict_probability(theta_0, theta_weights, x_sample):
    """
    Full hypothesis for logistic regression:
        h(x) = sigmoid(z)
    where z is computed by the linear function.
    """
    z = _compute_z(theta_0, theta_weights, x_sample)
    return _sigmoid(z)


# -------------------------------------------------------------
# Question 13 (from lab PDF):
# "Provide the complete, commented Python code for the CoreLogisticRegression class."
# -> Full trainable logistic regression model
# -------------------------------------------------------------
class CoreLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the model's hyperparameters.
        learning_rate : step size for gradient descent
        n_iterations  : number of passes over the full dataset
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta_0 = 0.0
        self.theta_weights = []
        self.cost_history = []

    def _sigmoid(self, z):
        """Sigmoid with numerical guards (same idea as global _sigmoid)."""
        if z > 700:
            return 1.0
        elif z < -700:
            return 0.0
        else:
            return 1.0 / (1.0 + math.exp(-z))

    def _compute_z(self, x_sample):
        """Computes z = theta_0 + sum(theta_i * x_i) for one sample."""
        z = self.theta_0
        for w, x in zip(self.theta_weights, x_sample):
            z += w * x
        return z

    def _predict_probability(self, x_sample):
        """Makes a probability prediction for a single sample."""
        z = self._compute_z(x_sample)
        return self._sigmoid(z)

    def _compute_cost(self, y_true, y_pred_probs):
        """
        Computes the Binary Cross-Entropy (log loss) cost.

        y_true       : list of true labels (0 or 1)
        y_pred_probs : list of predicted probabilities in [0,1]
        """
        m = len(y_true)
        if m == 0:
            return 0.0

        total_cost = 0.0
        epsilon = 1e-9  # to avoid log(0)

        for y, p in zip(y_true, y_pred_probs):
            # Clip probabilities to avoid exactly 0 or 1
            h = max(epsilon, min(1.0 - epsilon, p))
            cost_sample = -y * math.log(h) - (1 - y) * math.log(1 - h)
            total_cost += cost_sample

        return total_cost / m

    def _compute_gradients(self, X_data, y_true, y_pred_probs):
        """
        Computes gradients of the cost function for:
            - theta_0 (bias)
            - theta_weights (list of weights)
        using batch gradient descent.
        """
        m = len(y_true)
        n_features = len(self.theta_weights)

        grad_theta_0 = 0.0
        grad_theta_weights = [0.0] * n_features

        for i in range(m):
            error = y_pred_probs[i] - y_true[i]
            grad_theta_0 += error
            for j in range(n_features):
                grad_theta_weights[j] += error * X_data[i][j]

        # Average over all samples
        grad_theta_0 /= m
        for j in range(n_features):
            grad_theta_weights[j] /= m

        return grad_theta_0, grad_theta_weights

    def fit(self, X_data, y_data, verbose=True):
        """
        Trains the model using batch gradient descent.

        X_data : list of samples, each sample is [x1, x2, ...]
        y_data : list of true labels (0 or 1)
        """
        n_features = len(X_data[0])
        self.theta_0 = 0.0
        self.theta_weights = [0.0] * n_features
        self.cost_history = []

        for i in range(self.n_iterations):
            # 1. Predictions
            y_pred_probs = [self._predict_probability(x) for x in X_data]

            # 2. Cost
            cost = self._compute_cost(y_data, y_pred_probs)
            self.cost_history.append(cost)

            # 3. Gradients
            grad_theta_0, grad_theta_weights = self._compute_gradients(
                X_data, y_data, y_pred_probs
            )

            # 4. Parameter update
            self.theta_0 -= self.learning_rate * grad_theta_0
            for j in range(n_features):
                self.theta_weights[j] -= self.learning_rate * grad_theta_weights[j]

            # 5. Optional progress prints
            if verbose and self.n_iterations >= 10 and i % (self.n_iterations // 10) == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")

    def predict_proba(self, X_data):
        """
        Predicts probabilities for each sample in X_data.
        Returns a list of probabilities in [0,1].
        """
        return [self._predict_probability(x) for x in X_data]

    def predict(self, X_data, threshold=0.5):
        """
        Predicts class labels (0 or 1) based on the given threshold.
        """
        probabilities = self.predict_proba(X_data)
        return [1 if prob >= threshold else 0 for prob in probabilities]


# -------------------------------------------------------------
# Question 14 (from lab PDF):
# "How do we test this model? Provide the setup for a sample experiment."
# -> Simple experiment on 'hours studied' vs 'pass/fail'
# -------------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing CoreLogisticRegression ---")

    # 1. Create a simple dataset: hours studied
    X_train = [[1.0], [1.5], [2.0], [2.5], [4.5], [5.0], [5.5], [6.0]]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]  # > 4 hours => pass (1)

    # 2. Initialize and train the model
    model = CoreLogisticRegression(learning_rate=0.1, n_iterations=5000)
    print("Starting training...")
    model.fit(X_train, y_train, verbose=True)
    print("Training complete.")

    # 3. Print the final learned parameters
    print(f"\nFinal Bias (theta_0): {model.theta_0:.4f}")
    print(f"Final Weights (theta_1...): {model.theta_weights}")

    # 4. Make predictions on new, unseen data
    X_test = [[0.5], [3.0], [3.5], [7.0]]
    probs = model.predict_proba(X_test)
    labels = model.predict(X_test)

    print("\n--- Test Results ---")
    for i in range(len(X_test)):
        print(
            f"Input: {X_test[i][0]} hours | "
            f"Prob(Pass): {probs[i]:.4f} | "
            f"Prediction: {labels[i]}"
        )
