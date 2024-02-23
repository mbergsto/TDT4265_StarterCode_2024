import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    # Find mean and standarddeviation for the whole dataset
    mean = np.mean(X)
    std = np.std(X)
    print("Mean: ", mean, " \nStd: ", std)
    # Normalize the dataset
    X = (X - mean) / std
    # Add a column of ones to the data
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((X, ones))
    

    return X
 
def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    cross_entropy_error = - np.sum(targets * np.log(outputs)) / targets.shape[0]
    return cross_entropy_error


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 784  # Number of input nodes
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        #prev = self.I
        # Given code
        # for size in self.neurons_per_layer:
        #     prev += 1  # +1 for the bias trick - EDITED
        #     w_shape = (prev, size)
        #     print("Initializing weight to shape:", w_shape)
        #     w = np.zeros(w_shape)
        #     self.ws.append(w)
        #     prev = size


        #Edited
        if use_improved_weight_init:
            self.ws = [np.random.normal(0, 1/np.sqrt(self.I), (785, neurons_per_layer[0])), np.random.normal(0, 1/np.sqrt(neurons_per_layer[0]), (neurons_per_layer[0]+1, 10))]
        else:
            self.ws = [np.random.uniform(-1, 1, (785, neurons_per_layer[0])), np.random.uniform(-1, 1, (neurons_per_layer[0]+1, 10))]
        
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        # Dimensions. including bias:
        #I - Input Layer (785), J - Hidden Layer (65), 
        # K - Output Layer (10), N - Batch size 
        W_hidden = self.ws[0].T      #[64, 785]
        W_output = self.ws[1].T      #[10, 65] 
        
        self.z_hidden = W_hidden @ X.T                                                              #[64, 785] @ [785, N] = [64, N] 
        self.a_hidden = self.sigmoid(self.z_hidden)                                                 #[64, N]
        
        self.a_hidden = np.append(self.a_hidden, np.ones((1, self.a_hidden.shape[1])), axis=0)      #[65, N]
        
        self.z_output = W_output @ self.a_hidden                                                    #[10, 65] @ [65, N] = [10, N]
        yhat = np.exp(self.z_output) / np.sum(np.exp(self.z_output), axis=0)                        #[10, N]
        
        return yhat.T                                                                               #[N, 10]                    


    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer

        W_output = self.ws[1].T                         #[10, 65]
        W_output_tilde = W_output[:, : -1]              #[10, 64] # Remove the bias from the output layer
        N = X.shape[0]                                  #Batch size

        y_hat = outputs.T                               #[10, N]
        y = targets.T                                   #[10, N]

        delta_output = y_hat - y                                                                    #[10, N]        
        delta_hidden = (W_output_tilde.T @ delta_output) * self.sigmoid_derivative(self.z_hidden)        #[64, 10] @ [10, N] = [64, N]
        
        gradient_hidden = (delta_hidden @ X) / N                                                    #[64, N] @ [N, 785] = [64, 785]
        gradient_output = (delta_output @ self.a_hidden.T) / N                                      #[10, N] @ [N, 65] = [10, 65]

        self.grads[0] = gradient_hidden.T                                                           #[785, 64]
        self.grads[1] = gradient_output.T                                                           #[65, 10]    
              
        
        
        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def improved_sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.7159 * np.tanh(2/3 * z)

    def sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.use_improved_sigmoid:
            return 1.7159 * 2/3 * (1 - np.tanh(2/3 * z)**2)
        
        sigmoid = self.sigmoid(z)
        return sigmoid * (1 - sigmoid)

def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    num_examples = Y.shape[0]
    Y_one_hot = np.zeros((num_examples, num_classes))
    for i in range(num_examples):
        class_index = int(Y[i])
        Y_one_hot[i, class_index] = 1
    return Y_one_hot



def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)

    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
