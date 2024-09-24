import tkinter as tk
from tkinter import ttk
from prediction_model import PredictionModel
import sys


class AutoCompleteApp:
    def __init__(self, root, window = None):
        self.root = root
        self.root.title("Text Auto-Completion")
        self.root.geometry("400x100")
        if window:
            self.prediction_window = window
        else:
            self.prediction_window = 5
        self.model = PredictionModel('model_long_data.pt', self.prediction_window)
        
        self.text_var = tk.StringVar()
        self.entry = ttk.Entry(root, textvariable=self.text_var, width=50)
        self.entry.pack(pady=20)

        self.suggestion_label = ttk.Label(root, text="", foreground="gray")
        self.suggestion_label.pack()

        self.entry.bind('<KeyRelease>', self.on_key_release)
        self.entry.bind('<Tab>', self.on_tab)
        self.entry.bind('<Return>', self.on_enter)

        self.suggested_text = ""

    def on_key_release(self, event):
        if event.keysym == "Tab" or event.keysym == "Return":
            return

        current_text = self.text_var.get()
        if current_text.strip():
            self.suggested_text = self.model(current_text)
            self.suggestion_label.config(text=self.suggested_text)
        else:
            self.suggestion_label.config(text="")
            self.suggested_text = ""

    def on_tab(self, event):
        current_text = self.text_var.get()
        if self.suggested_text:
            self.text_var.set(current_text + self.suggested_text)
            self.suggestion_label.config(text="")
            self.entry.icursor(len(self.text_var.get()))
        return "break"

    def on_enter(self, event):
        self.text_var.set("")
        self.suggestion_label.config(text="")
        self.suggested_text = ""
        return "break"  

if __name__ == "__main__":
    try:
        window = int(sys.argv[1])
    except:
        window = None
    root = tk.Tk()
    app = AutoCompleteApp(root, window)
    root.mainloop()
