import logging
import os
import pathlib

# https://github.com/Android-for-Python/camera4kivy#camera-provider
from kivy import kivy_options
providers= list(kivy_options['camera'])
providers.remove('opencv')  # low resolution on Linux
kivy_options['camera'] = tuple(providers)

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, NoTransition

from app import androidperm
from app.screens.main import MainScreen
from app.screens.results import ResultScreen
import app.const


__version__ = '0.5.3'

DEBUG = os.getenv('DEBUG')
ROOT_DIRPATH = pathlib.Path(__file__).parent

lgr = logging


class BidlyApp(App):

    def build(self):
        self.bidly = ScreenManager(transition=NoTransition())
        self.main_screen = MainScreen(name=app.const.MAIN_SCREEN)
        self.bidly.add_widget(self.main_screen)
        self.bidly.add_widget(ResultScreen(name=app.const.RESULT_SCREEN))
        return self.bidly

    def on_start(self):
        self.dont_gc = androidperm.AndroidPermissions(self.start_app)

    def on_stop(self):
        self.main_screen.camera_square.camera.disconnect_camera()

    def start_app(self):
        self.dont_gc = None
        Clock.schedule_once(self.connect_camera)

    def connect_camera(self, dt):
        main_screen = self.bidly.get_screen(app.const.MAIN_SCREEN)
        main_screen.camera_square.camera.connect()
        main_screen.button_box.camera_button.disabled = False


if __name__ == '__main__':
    BidlyApp().run()
