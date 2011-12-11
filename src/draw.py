from matplotlib import pyplot as plt


def draw_plots(elements, legend=None):
    """
    Draw elements
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yscale('log')
    for element in elements:
        ax.plot(element)
    if legend:
        ax.legend(legend)
    fig.show()


def error(ground_truth, S):
    """
    Calculate the error
    """
    err = np.sqrt((ground_truth - S) ** 2)
    return err
