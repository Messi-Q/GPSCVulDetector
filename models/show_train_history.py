import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    """
    显示训练过程

    参数：
        train_history - 训练结果存储的参数位置
        train - 训练数据的执行结果
        validation - 验证数据的执行结果
    """
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


