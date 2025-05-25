import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os
import sys, os, subprocess, tkinter as tk
from tkinter import ttk

# ---------- helper that works frozen / unfrozen ----------
def resource_path(rel_path: str) -> str:
    """Return absolute path to resource inside bundle or next to script"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(__file__))  # _MEIPASS exists only when frozen
    return os.path.join(base, rel_path)

class ChooseLanguageWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(height=False, width=False)
        self.root.title("Copilot Gamer")
        icon_path = os.path.join(os.path.dirname(__file__), "copilot.png")
        self.root.iconphoto(False, tk.PhotoImage(file=icon_path))

        self.cogamer_process = None  # To store the subprocess reference

        # Pre-built Gemini TTS voices (speech-generation guide, May 2025)
        self.voices = {
            "Zephyr": "Zephyr",  # bright
            "Puck": "Puck",  # upbeat
            "Charon": "Charon",  # informative
            "Kore": "Kore",  # firm
            "Fenrir": "Fenrir",  # excitable
            "Leda": "Leda",  # youthful
            "Orus": "Orus",  # firm
            "Aoede": "Aoede",  # breezy
            "Callirhoe": "Callirhoe",  # easy-going
            "Autonoe": "Autonoe",  # bright
            "Enceladus": "Enceladus",  # breathy
            "Iapetus": "Iapetus",  # clear
            "Umbriel": "Umbriel",  # easy-going
            "Algieba": "Algieba",  # smooth
            "Despina": "Despina",  # smooth
            "Erinome": "Erinome",  # clear
            "Algenib": "Algenib",  # gravelly
            "Rasalgethi": "Rasalgethi",  # informative
            "Laomedeia": "Laomedeia",  # upbeat
            "Achernar": "Achernar",  # soft
            "Alnilam": "Alnilam",  # firm
            "Schedar": "Schedar",  # even
            "Gacrux": "Gacrux",  # mature
            "Pulcherrima": "Pulcherrima",  # forward
            "Achird": "Achird",  # friendly
            "Zubenelgenubi": "Zubenelgenubi",  # casual
            "Vindemiatrix": "Vindemiatrix",  # gentle
            "Sadachbia": "Sadachbia",  # lively
            "Sadaltager": "Sadaltager",  # knowledgeable
            "Sulafar": "Sulafar",  # warm
        }
        self.basic_voice = tk.StringVar(value=self.voices["Zephyr"])

        self.label1 = ttk.Label(self.root, text="Hello Everybody! Cogamer's greetings!\n")
        self.label1.pack()
        self.label2 = ttk.Label(self.root, text=f"Choose a copilot voice")
        self.label2.pack(padx=10, pady=6, anchor='w')


        COLS = 3                               # how many columns you want
        voice_frame = ttk.Frame(self.root)
        voice_frame.pack(padx=10, pady=6, fill="x")

        for idx, voice in enumerate(self.voices):
            r = idx // COLS                    # row index
            c = idx %  COLS                    # column index
            ttk.Radiobutton(
                voice_frame,
                text=voice,
                value=voice,
                variable=self.basic_voice,
                command=self.select
            ).grid(row=r, column=c, sticky="w", padx=5, pady=3)

        # optional: widen the window a bit
        self.root.geometry("340x400")

        btn_ok = ttk.Button(self.root, text="OK", command=self.start_cogamer)
        btn_ok.place(relx=0, rely=0.85)
        btn_exit = ttk.Button(self.root, text="Exit", command=self.exit_from_app)
        btn_exit.place(relx=0.4, rely=0.85)

        # Ensure cogamer_process is terminated when the window is closed via 'X' button
        self.root.protocol("WM_DELETE_WINDOW", self.exit_from_app)

    def run(self):
        self.root.mainloop()

    def select(self):
        self.label2.config(text=f"Choose a copilot voice")

    def exit_from_app(self):
        if self.cogamer_process and self.cogamer_process.poll() is None:  # Check if process exists and is running
            print("Terminating cogamer.py process...")
            self.cogamer_process.terminate()  # Terminate the process
            try:
                self.cogamer_process.wait(timeout=5)  # Wait for graceful termination
            except subprocess.TimeoutExpired:
                print("cogamer.py did not terminate gracefully, killing it.")
                self.cogamer_process.kill()  # Force kill if it doesn't terminate

        print("App has been closed")
        self.root.quit()
        self.root.destroy()

    def start_cogamer(self):
        if self.cogamer_process and self.cogamer_process.poll() is None:
            print("Cogamer is already running, stopping it first...")
            self.cogamer_process.terminate()
        # Where is the executable?
        if getattr(sys, "frozen", False):             # running inside PyInstaller bundle
            cogamer_exe = resource_path("cogamer")    # we’ll embed it with --add-binary
            cmd = [cogamer_exe]
        else:                                         # dev mode
            cogamer_exe = os.path.join(os.path.dirname(__file__), "cogamer.py")
            cmd = [sys.executable, cogamer_exe]

        chosen_voice = self.basic_voice.get()
        self.cogamer_process = subprocess.Popen(cmd + [chosen_voice])


if __name__ == "__main__":
    window = ChooseLanguageWindow()
    window.run()