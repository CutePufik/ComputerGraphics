import tkinter as tk

from windows.task1_window import Task1Window



class Lab3:
    def __init__(self, root: tk.Tk):
        self.back_ground = "#333"

        self.root = root
        self.root.title("Lab3")
        self.root.configure(bg=self.back_ground)
        self.root.geometry("300x110+1+1")

        # task1
        self.task1_button = tk.Button(
            root,
            text="task1",
            command=self.task1,
            bg="#555",
            fg="white",
            width=100,
        )


        self.task1_button.pack(pady=5, padx=5)


    def task1(self):
        child = tk.Tk()
        task1_window = Task1Window(root=child, parent=self)




if __name__ == "__main__":
    root = tk.Tk()
    app = Lab3(root)
    root.mainloop()