import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os

class ChooseLanguageWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("300x300")
        self.root.resizable(height=False, width=False)
        self.root.title("Copilot Gamer")
        icon_path = os.path.join(os.path.dirname(__file__), "copilot.png")
        self.root.iconphoto(False, tk.PhotoImage(file=icon_path))

        self.voices = {
            "Puck": "Puck",
            "Charon": "Charon",
            "Kore": "Kore",
            "Fenrir": "Fenrir",
            "Aoede": "Aoede"
        }
        self.basic_voice = tk.StringVar(value=self.voices["Fenrir"])

        self.label1 = ttk.Label(self.root, text="Hello Everybody! Cogamer's greetings!\n")
        self.label1.pack()
        self.label2 = ttk.Label(self.root, text=f"Choose a copilot voice. Now '{self.basic_voice.get()}' has chosen")
        self.label2.pack(padx=10, pady=6, anchor='w')

        for voice in self.voices.keys():
            btn = ttk.Radiobutton(
                self.root,
                text=self.voices[voice],
                value=self.voices[voice],
                variable=self.basic_voice,
                command=self.select
            )
            btn.pack(padx=20, pady=6, anchor='w')

        btn_ok = ttk.Button(self.root, text="OK", command=self.start_cogamer)
        btn_ok.place(relx=0.4, rely=0.85)
        btn_exit = ttk.Button(self.root, text="Exit", command=self.exit_from_app)
        btn_exit.place(relx=0.7, rely=0.85)

    def run(self):
        self.root.mainloop()

    def select(self):
        self.label2.config(text=f"Choose a copilot voice. Now '{self.basic_voice.get()}' has chosen")

    def exit_from_app(self):
        self.root.quit()
        self.root.destroy()
        print("App has been closed")

    def start_cogamer(self):
        chosen_voice = self.basic_voice.get()
        self.exit_from_app()  # Ensure GUI cleans up fully
        print(f"Voice {chosen_voice} has chosen")
        # Launch cogamer.py in a separate process, passing the chosen voice as an argument
        subprocess.Popen([sys.executable, "cogamer.py", chosen_voice])

if __name__ == "__main__":
    window = ChooseLanguageWindow()
    window.run()
