from tkinter import *
from tkinter import simpledialog


class MyDialog(simpledialog.Dialog):
    def body(self, master):
        self.geometry("300x300")
        Label(master, text="Enter your search string text:").grid(row=0)

        self.e1 = Entry(master)
        self.e1.grid(row=0, column=1)
        return self.e1  # initial focus

    def apply(self):
        first = self.e1.get()
        self.result = first
