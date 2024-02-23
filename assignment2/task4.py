import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy

def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 5
    learning_rate = .02  #Edited for 3c
    batch_size = 32
    neurons = [32, 64, 128]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)


    for neuron in neurons:
        model = SoftmaxModel(
            [neuron, 10],
            use_improved_sigmoid,
            use_improved_weight_init,
            use_relu)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )

        train_history, val_history = trainer.train(num_epochs)
        #print accuracy and loss
        print("Neuron: ", neuron)
        print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:", cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

        # Plot loss
        # plt.subplot(1, 2, 1)
        # utils.plot_loss(train_history["loss"], neuron, npoints_to_average=10)
        # utils.plot_loss(val_history["loss"], neuron, npoints_to_average=10)
        # plt.ylabel("Cross Entropy Loss")
        # plt.xlabel("Number of Training Steps")
        # plt.ylim([0.0, .5])
        # #plt.legend()
        # plt.grid(True)
        

        plt.subplot(1, 1, 1)
        utils.plot_loss(val_history["accuracy"], neuron)
        plt.ylabel("Accuracy")
        plt.xlabel("Number of Training Steps")
        plt.ylim([0.85, 1.00])
        plt.legend()
        plt.grid(True)
    
    plt.savefig("task4ab.png")
    plt.show()




if __name__ == "__main__":
    main()
