"""SoftmaxRegressor

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import numpy as np


from belt.supervised.models import supervised_model_registry


@supervised_model_registry.register("SoftmaxRegressor")
class SoftmaxRegressor:
    def __init__(
        self,
        input_dim=28 * 28,
        num_classes=10,
        weight_scale=0.001,
        seed=1024,
        relu_logits=True,
    ):
        """
        A single layer softmax regression linear -> ReLU -> Softmax.

        Parameters
        ----------
        input_dim : int
            number of input features/dimesions
        num_classes : int
            output classes
        weight_scale : float
            random weight init scale factor
        seed : int
            weight init random seed
        relu_logits : bool
            Whether to apply ReLU before softmax
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.weight_scale = weight_scale
        self.seed = seed
        self.relu_logits = relu_logits

        self.weights = dict()
        self.gradients = dict()

        self._weight_init()

    def _weight_init(self):
        """
        init weights (input_dim, num_classes) to small random values
        and the gradients to zeros
        """
        np.random.seed(self.seed)
        self.weights["W1"] = self.weight_scale * np.random.randn(
            self.input_dim, self.num_classes
        )
        self.gradients["W1"] = np.zeros((self.input_dim, self.num_classes))

    def forward(self, X, y, mode="train"):
        """
        Compute loss and gradients using softmax with vectorization

        Parameters
        ----------
        X : np.ndarray of shape (N, input_dim)
            Batch of input features.
        y : np.ndarray of shape (N,)
            True class labels for the batch.
        mode : str
            'train' computes and stores gradients

        Returns
        -------
        loss : float
            cross entropy loss for the batch.
        accuracy : float
            batch classification accuracy
        """
        # x -> Wx -> ReLU (optional) -> Softmax -> Cross-Entropy Loss
        w = self.weights["W1"]
        logits = X @ w

        if self.relu_logits:
            activated = self.ReLU(logits)
        else:
            activated = logits

        probs = self.softmax(activated)
        loss = self.cross_entropy_loss(probs, y)
        accuracy = self.compute_accuracy(probs, y)

        if mode != "train":
            return loss, accuracy

        # One-hot encode the true labels
        y_true = np.zeros((y.size, probs.shape[1]))
        y_true[np.arange(y.size), y] = 1

        # dLoss / dSoftmax
        dL_dS = np.subtract(probs, y_true)

        if self.relu_logits:
            # dReLU * dL/dSoftmax
            dR = self.ReLU_dev(logits)
            delta = dL_dS * dR
        else:
            delta = dL_dS

        # dL / dInputLayer; take mean since loss is an average
        dW = X.T @ delta
        dW = dW / X.shape[0]

        self.gradients["W1"] = dW
        return loss, accuracy

    def predict(self, X):
        """
        Predict class labels

        Parameters
        ----------
        X : np.ndarray of shape (N, input_dim)

        Returns
        -------
        preds : np.ndarray of shape (N,)
            predicted class indices
        """
        w = self.weights["W1"]
        logits = X @ w
        activated = self.ReLU(logits) if self.relu_logits else logits
        probs = self.softmax(activated)
        return np.argmax(probs, axis=1)

    def state_dict(self):
        """
        Return a copy of the model's weights
        """
        return {k: v.copy() for k, v in self.weights.items()}

    def softmax(self, scores):
        """
        Compute softmax probabilities from scores

        Parameters
        ----------
        scores : np.ndarray of shape (N, num_classes)
            logits

        Returns
        -------
        prob : np.ndarray of shape (N, num_classes)
            softmax probabilities
        """
        # Subtract max score per row to prevent exponent overflow
        norm_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(norm_scores)
        prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        mean cross entropy loss

        Parameters
        ----------
        x_pred : np.ndarray of shape (N, num_classes)
            softmax probabilities
        y : np.ndarray of shape (N,)
            ground truth labels

        Returns
        -------
        loss : float
            mean cross entropy loss over the batch.
        """
        pred_class_probs = x_pred[np.arange(len(y)), y]
        # loss = mean(-log(P))
        loss = np.mean(-np.log(pred_class_probs))
        return loss

    def compute_accuracy(self, x_pred, y):
        """
        per-batch classification accuracy

        Parameters
        ----------
        x_pred : np.ndarray of shape (N, num_classes)
            softmax probabilities
        y : np.ndarray of shape (N,)
            ground truth labels

        Returns
        -------
        acc : float
            percent of correct labels
        """
        # Max probability in each row is the predicted class
        preds = np.argmax(x_pred, axis=1)
        acc = np.mean(preds == y)
        return acc

    def ReLU(self, X):
        """
        element-wise ReLU activation

        Parameters
        ----------
        X : np.ndarray of shape (N, layer_size)
            activations

        Returns
        -------
        out : np.ndarray of shape (N, layer_size)
            negative values zeroed
        """
        # return 0 if negative
        out = [[z if z > 0 else 0 for z in x] for x in X]
        return np.array(out)

    def ReLU_dev(self, X):
        """
        element-wise gradient of ReLU

        Parameters
        ----------
        X : np.ndarray of shape (N, layer_size)
            Input activations

        Returns
        -------
        out : np.ndarray of shape (N, layer_size)
            1 where X > 0, 0 elsewhere.
        """
        # return 1 if positive, 0 if negative
        out = [[1 if z > 0 else 0 for z in x] for x in X]
        return np.array(out)
