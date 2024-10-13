import tkinter as tk
from tkinter import messagebox, Toplevel

from Service import train

class NeuralNetworkWindow:
    def __init__(self, master):
        self.master = Toplevel(master)
        self.master.title("Neural Network")
        self.master.geometry("512x512")

        # Create frames for different areas
        self.epochs_frame = tk.Frame(self.master)
        self.epochs_frame.place(relx=0.5, rely=0.3, anchor='center')

        self.learning_frame = tk.Frame(self.master)
        self.learning_frame.place(relx=0.5, rely=0.5, anchor='center')

        self.actions_frame = tk.Frame(self.master)
        self.actions_frame.place(relx=0.5, rely=0.7, anchor='center')

        # Epochs input
        tk.Label(self.epochs_frame, text="Epochs:").pack(side='left')
        self.epochs_entry = tk.Entry(self.epochs_frame)
        self.epochs_entry.pack(side='left')

        # Buttons for learning processes
        learn_new_button = tk.Button(self.learning_frame, text="Train from Scratch", command=self.train_from_scratch)
        learn_new_button.pack(side='left', padx=5)

        learn_button = tk.Button(self.learning_frame, text="Retrain", command=self.retrain)
        learn_button.pack(side='left', padx=5)

        # Output scales button
        output_scales_button = tk.Button(self.actions_frame, text="Output Scales", command=self.output_scales)
        output_scales_button.pack(side='left', padx=5)

        # Buttons for saving and displaying weights
        save_weights_button = tk.Button(self.actions_frame, text="Save Weights", command=self.save_weights)
        save_weights_button.pack(side='left', padx=5)

        show_weights_button = tk.Button(self.actions_frame, text="Show Weights", command=self.show_weights)
        show_weights_button.pack(side='left', padx=5)

    def train_from_scratch(self):
        if self.confirm_action("Are you sure you want to start learning from scratch?"):
            epochs = self.get_epochs()
            if epochs is not None:
                print(f"Learning from scratch for {epochs} epochs.")
                train(epochs, True)

    def retrain(self):
        if self.confirm_action("Are you sure you want to start learning without zeroing weights?"):
            epochs = self.get_epochs()
            if epochs is not None:
                print(f"Learning for {epochs} epochs without zeroing weights.")
                train(epochs, False)


    def get_epochs(self):
        try:
            epochs = int(self.epochs_entry.get())
            return epochs
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid number of epochs.")
            return None

    def output_scales(self):
        print("Outputting scales...")

    def save_weights(self):
        print("Saving weights to file...")

    def show_weights(self):
        new_window = Toplevel(self.master)
        new_window.title("Weights")
        tk.Label(new_window, text="Weights go here").pack()

    def confirm_action(self, message):
        return messagebox.askyesno("Confirm Action", message)
