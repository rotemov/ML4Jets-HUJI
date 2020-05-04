import matplotlib.pyplot as plt


class Plotter:
    """
    Helper Object to easily make plots of given signal and background event objects.
    Uses matplotlib.pyplot
    """

    def __init__(self, signal_events, background_events):
        """
        :param signal_events: A list of signal Event objects
        :param background_events: A list of background Event objects
        """
        self.signal_events = signal_events
        self.background_events = background_events

    def plot_histogram(self, func, x_label, y_label):
        """
        Plot a histogram of the signal and background events.

        :param func: The function to map the events to before plotting
        :param x_label: x axis label
        :param y_label: y axis label
        """
        plt.figure()
        plt.hist(self._flatten(map(func, self.background_events)), bins=50, facecolor='b', label='background')
        plt.hist(self._flatten(map(func, self.signal_events)), bins=50, facecolor='r', label='signal')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper right')
        plt.show()

    def plot_scatter(self, xfunc, yfunc, x_label, y_label):
        """
        Plot a scatter plot of the signal and background events.

        :param xfunc: The function to map the events to for the x axis
        :param yfunc: The function to map the events to for the y axis
        :param x_label: x axis label
        :param y_label: y axis label
        """
        plt.figure()
        plt.scatter(self._flatten(map(xfunc, self.background_events)), self._flatten(map(yfunc, self.background_events)), c='b', label='background')
        plt.scatter(self._flatten(map(xfunc, self.signal_events)), self._flatten(map(yfunc, self.signal_events)), c='r', label='signal')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper left')
        plt.show()

    def _flatten(self, l):
        """
        Flatten the list l if it contains more lists

        :param l: A list, can be a list of lists
        :return: A list of numbers
        """
        flat_list = []
        for item in l:
            if isinstance(item, list):
                flat_list += item
            else:
                flat_list += [item]
        return flat_list
