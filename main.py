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

from app import androidperm
from app.screens.main import MainScreen


__version__ = '0.4.2'

DEBUG = os.getenv('DEBUG')
ROOT_DIRPATH = pathlib.Path(__file__).parent

lgr = logging


class BidlyApp(App):
    def build(self):
        self.bidly = MainScreen()
        return self.bidly

    def on_start(self):
        self.dont_gc = androidperm.AndroidPermissions(self.start_app)

    def on_stop(self):
        self.bidly.deal_box.camera_square.camera.disconnect_camera()

    def start_app(self):
        self.dont_gc = None
        Clock.schedule_once(self.connect_camera)

    def connect_camera(self, dt):
        self.bidly.deal_box.camera_square.camera.connect()


if __name__ == '__main__':
    BidlyApp().run()
