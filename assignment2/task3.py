import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!
    
    use_improved_weight_init=True
    
    
    wei_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    wei_trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        wei_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    wei_train_history, wei_val_history = wei_trainer.train(num_epochs)
    
    use_improved_sigmoid=True
    use_improved_weight_init=True
    
    
    sig_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    sig_trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        sig_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    sig_train_history, sig_val_history = sig_trainer.train(num_epochs)
    
    use_momentum=True
    use_improved_sigmoid=True
    use_improved_weight_init=True
    
    learning_rate=0.03
    mom_model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    mom_trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        mom_model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    mom_train_history, mom_val_history = mom_trainer.train(num_epochs)
    
    
    
    
        # Plot loss for first model (task 2c)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    # plt.ylim([0., 5.5])
    plt.ylim([0., .5])
    
    
    utils.plot_loss(train_history["loss"],"Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    
    utils.plot_loss(wei_train_history["loss"],"improved Weight Training Loss", npoints_to_average=10)
    utils.plot_loss(wei_val_history["loss"], "improved Weight Validation Loss")
    
    
    utils.plot_loss(mom_train_history["loss"],"Momentum Training Loss", npoints_to_average=10)
    utils.plot_loss(mom_val_history["loss"], "Momentum Validation Loss")
    
    
    utils.plot_loss(sig_train_history["loss"],"improved sigmoid Training Loss", npoints_to_average=10)
    utils.plot_loss(sig_val_history["loss"], "improved sigmoid Validation Loss")
  
    
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    
    
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.90, .99])
    
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    
    utils.plot_loss(wei_train_history["accuracy"], "improved Weight Training Accuracy")
    utils.plot_loss(wei_val_history["accuracy"], "improved Weight Validation Accuracy")
    
    
    utils.plot_loss(mom_train_history["accuracy"], "Momentum Training Accuracy")
    utils.plot_loss(mom_val_history["accuracy"], "Momentum Validation Accuracy")
    
    utils.plot_loss(sig_train_history["accuracy"], "improved sigmoid Training Accuracy")
    utils.plot_loss(sig_val_history["accuracy"], "improved sigmoid Validation Accuracy")
    
    
    
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("Task3wei.png")
    
    
    
    
    
    
    
    # shuffle_data = False
    # model_no_shuffle = SoftmaxModel(
    #     neurons_per_layer,
    #     use_improved_sigmoid,
    #     use_improved_weight_init)
    # trainer_shuffle = SoftmaxTrainer(
    #     momentum_gamma, use_momentum,
    #     model_no_shuffle, learning_rate, batch_size, shuffle_data,
    #     X_train, Y_train, X_val, Y_val,
    # )
    # train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
    #     num_epochs)
    # shuffle_data = True

    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history["loss"],
    #                 "Task 2 Model", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history_no_shuffle["loss"], "Task 2 Model - No dataset shuffling", npoints_to_average=10)
    # plt.ylim([0, .4])
    # plt.subplot(1, 2, 2)
    # plt.ylim([0.85, .95])
    # utils.plot_loss(
    #     val_history_no_shuffle["accuracy"], "Task 2 Model - No Dataset Shuffling")
    # utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    # plt.ylabel("Validation Accuracy")
    # plt.legend()
    # # plt.show()
    # plt.savefig("task3.png")
