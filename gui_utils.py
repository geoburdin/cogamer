import tkinter as tk
from tkinter import ttk
from cogamer import cogamer


class ChooseLanguageWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("300x300")
        self.root.resizable(height=False, width=False)
        self.root.title("Copilot Gamer")  # set a window title
        self.root.iconphoto(False, tk.PhotoImage(file="copilot.png"))
        self.voices = {"Puck": "Puck", "Charon": "Charon", "Kore": "Kore", "Fenrir": "Fenrir", "Aoede": "Aoede"}
        self.basic_voice = tk.StringVar(value=self.voices["Fenrir"])
        self.label1 = ttk.Label(text="Hello Everybody! Cogamer's greetings!\n")  # create a text label
        self.label1.pack()  # place a label in the window
        self.label2 = ttk.Label(text=f"Choose a copilot voice. Now '{self.basic_voice.get()}' has chosen")
        self.label2.pack(padx=10, pady=6, anchor='w')

        position = {"padx": 20, "pady": 6, "anchor": "w"}
        for voice in self.voices.keys():
            self.voice_btn = ttk.Radiobutton(text=self.voices[voice],
                                             value=self.voices[voice],
                                             variable=self.basic_voice,
                                             command=self.select)
            self.voice_btn.pack(**position)

        btn_ok = ttk.Button(text="OK", command=lambda: self.start_cogamer(voice=self.basic_voice.get()))
        btn_ok.place(relx=0.4, rely=0.85)
        btn_exit = ttk.Button(text="Exit", command=lambda: self.exit_from_app(root=self.root))
        btn_exit.place(relx=0.7, rely=0.85)

    def run(self):
        self.root.mainloop()

    def select(self):
        self.label2.config(text=f"Choose a copilot voice. Now '{self.basic_voice.get()}' has chosen")

    def exit_from_app(self, root):
        root.destroy()  # closed window and app
        print("App has been closed")

    def start_cogamer(self, voice='Fenrir'):
        self.root.destroy()
        self.root.quit()
        tk._default_root = None
        print(f"Voice {voice} has chosen")
        cogamer(chosen_voice=voice)


if __name__ == "__main__":
    choose_language_window = ChooseLanguageWindow()
    choose_language_window.run()
