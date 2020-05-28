import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    """
    show the training process

    parameter：
        train_history: the trained parameter
        train: training result
        validation: validation result
    """
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


