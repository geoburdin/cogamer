import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os
import signal

class ChooseLanguageWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("400x400")
        self.root.resizable(False, False)
        self.root.title("AI Gaming Assistant")
        icon_path = os.path.join(os.path.dirname(__file__), "copilot.png")
        self.root.iconphoto(False, tk.PhotoImage(file=icon_path))

        self.voices = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]
        self.basic_voice = tk.StringVar(value="Fenrir")
        self.last_voice = None
        self.process = None

        ttk.Label(self.root, text="Cogamer's running!\n").pack()
        self.label2 = ttk.Label(
            self.root,
            text=f"Choose a copilot voice. Now - '{self.basic_voice.get()}'"
        )
        self.label2.pack(padx=10, pady=6, anchor='w')

        for v in self.voices:
            ttk.Radiobutton(
                self.root,
                text=v,
                value=v,
                variable=self.basic_voice,
                command=self._update_label
            ).pack(padx=20, pady=6, anchor='w')

        ttk.Button(self.root, text="Start/Restart", command=self.start_cogamer)\
            .place(relx=0.4, rely=0.85)
        ttk.Button(self.root, text="Exit", command=self.exit_from_app)\
            .place(relx=0.7, rely=0.85)

    def run(self):
        self.root.mainloop()

    def _update_label(self):
        self.label2.config(
            text=f"Choose a copilot voice. Now - '{self.basic_voice.get()}'"
        )

    def exit_from_app(self):
        if self.process and self.process.poll() is None:
            print("Exiting: killing cogamer…")
            self._kill_process()
        self.root.quit()
        self.root.destroy()
        print("App has been closed")

    def _kill_process(self):
        try:
            self.process.terminate()
            print(f"Killed process PID {self.process.pid}")
        except Exception as e:
            print("Error killing process:", e)
        finally:
            self.process = None

    def start_cogamer(self):
        chosen = self.basic_voice.get()
        if self.process and self.process.poll() is None:
            self._kill_process()

        if getattr(sys, "frozen", False):
            cogamer_exe = os.path.join(
                os.path.dirname(sys.executable),
                "cogamer" if sys.platform != "win32" else "cogamer.exe"
            )
            cmd = [cogamer_exe, chosen]
        else:
            cogamer_py = os.path.join(os.path.dirname(__file__), "cogamer.py")
            cmd = [sys.executable, cogamer_py, chosen]

        print("Launching:", " ".join(cmd))
        self.process = subprocess.Popen(cmd)


if __name__ == "__main__":
    ChooseLanguageWindow().run()
