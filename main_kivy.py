import os

os.environ["KIVY_LOG_MODE"] = "PYTHON"

import logging
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.logger import Logger, LOG_LEVELS

import cogamer

Window.size = (200, 200)
logging.basicConfig(level=logging.INFO)
Logger.setLevel(LOG_LEVELS["info"])


class VoiceLayout(BoxLayout):
    def __init__(self, **kw):
        super().__init__(**kw)
        box = BoxLayout(orientation="vertical")
        self.orientation = 'vertical'
        l1 = Label(text='Choose voice for Cogamer')
        button_puck = Button(text='Puck')
        button_puck.bind(on_press=self.voice_puck_choosen)
        button_charon = Button(text='Charon')
        button_charon.bind(on_press=self.voice_charon_choosen)
        button_kore = Button(text='Kore')
        button_kore.bind(on_press=self.voice_kore_choosen)
        button_fenrir = Button(text='Fenrir')
        button_fenrir.bind(on_press=self.voice_fenrir_choosen)
        button_aoede = Button(text='Aoede')
        button_aoede.bind(on_press=self.voice_aoede_choosen)
        self.add_widget(l1)
        self.add_widget(button_puck)
        self.add_widget(button_charon)
        self.add_widget(button_kore)
        self.add_widget(button_fenrir)
        self.add_widget(button_aoede)

    def voice_puck_choosen(self, instance):
        global voice_for_cogamer
        voice_for_cogamer = 'Puck'
        App.get_running_app().stop()
        Window.close()
        return 'Puck'

    def voice_charon_choosen(self, instance):
        global voice_for_cogamer
        voice_for_cogamer = 'Charon'
        App.get_running_app().stop()
        Window.close()
        return 'Charon'

    def voice_kore_choosen(self, instance):
        global voice_for_cogamer
        voice_for_cogamer = 'Kore'
        App.get_running_app().stop()
        Window.close()
        return 'Kore'

    def voice_fenrir_choosen(self, instance):
        global voice_for_cogamer
        voice_for_cogamer = 'Fenrir'
        App.get_running_app().stop()
        Window.close()
        return 'Fenrir'

    def voice_aoede_choosen(self, instance):
        global voice_for_cogamer
        voice_for_cogamer = 'Aoede'
        App.get_running_app().stop()
        Window.close()
        return 'Aoede'


class Cogamer(App):
    def build(self):
        return VoiceLayout()


if __name__ == '__main__':
    voice_for_cogamer = 'not_chosen'
    Cogamer().run()

    if voice_for_cogamer != 'not_chosen':
        print(f'Voice {voice_for_cogamer} has chosen')
        cogamer.cogamer(chosen_voice=voice_for_cogamer)
    else:
        print(f'Voice has not chosen')
