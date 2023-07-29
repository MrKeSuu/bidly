import logging
import os
import pathlib

# https://github.com/Android-for-Python/camera4kivy#camera-provider
from kivy import kivy_options
providers= list(kivy_options['camera'])
providers.remove('opencv')
kivy_options['camera'] = tuple(providers)

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager

from app import androidperm
from app.screens.main import MainScreen
from app.screens.results import ResultScreen


__version__ = '0.4.2'

DEBUG = os.getenv('DEBUG')
ROOT_DIRPATH = pathlib.Path(__file__).parent

lgr = logging


class BidlyApp(App):
    MAIN_SCREEN = 'main'
    RESULT_SCREEN = 'results'

    def build(self):
        self.bidly = ScreenManager()
        self.bidly.add_widget(MainScreen(name=self.MAIN_SCREEN))
        self.bidly.add_widget(ResultScreen(name=self.RESULT_SCREEN))
        return self.bidly

    def on_start(self):
        self.dont_gc = androidperm.AndroidPermissions(self.start_app)

    def on_stop(self):
        main_screen = self.bidly.get_screen(self.MAIN_SCREEN)
        main_screen.deal_box.camera_square.camera.disconnect_camera()

    def start_app(self):
        self.dont_gc = None
        Clock.schedule_once(self.connect_camera)

    def connect_camera(self, dt):
        main_screen = self.bidly.get_screen(self.MAIN_SCREEN)
        main_screen.deal_box.camera_square.camera.connect()


if __name__ == '__main__':
    BidlyApp().run()
