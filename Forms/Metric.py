import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Frame, Button
import matplotlib.backends.backend_tkagg


class PlotApp:
    def __init__(self, master, data):
        self.master = master
        self.data = data

        # Create a button to display the plots
        self.button = Button(master, text="Plot Graphs", command=self.show_plots)
        self.button.pack()

    def show_plots(self):
        # Create a new window for the plots
        plot_window = Tk()
        plot_window.title("Graphs")

        # Plot the graphs based on the data
        epochs = self.data.index

        # Loss
        plt.subplot(221)
        plt.plot(epochs, self.data['loss'], label='Loss', color='blue')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.grid()

        # Accuracy
        plt.subplot(222)
        plt.plot(epochs, self.data['accuracy'], label='Accuracy', color='green')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.grid()

        # Precision
        plt.subplot(223)
        plt.plot(epochs, self.data['precision'], label='Precision', color='orange')
        plt.title('Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.grid()

        # Recall
        plt.subplot(224)
        plt.plot(epochs, self.data['recall'], label='Recall', color='red')
        plt.title('Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.grid()

        # Adjust layout of the plots
        plt.tight_layout()

        # Embed the plots in Tkinter
        canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(plt.gcf(), master=plot_window)
        canvas.get_tk_widget().pack()
        canvas.draw()

        plot_window.mainloop()


def main():
    # Read data from CSV file
    file_name = 'data.csv'  # Specify the name of your CSV file
    data = pd.read_csv(file_name, sep=',', header=None, names=['loss', 'accuracy', 'precision', 'recall'])

    # Create the main Tkinter window
    root = Tk()
    root.title("Graphs from CSV")

    app = PlotApp(root, data)
    root.mainloop()


if __name__ == '__main__':
    main()
