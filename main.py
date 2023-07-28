import logging
import os
import pathlib

# https://github.com/Android-for-Python/camera4kivy#camera-provider
from kivy import kivy_options
providers= list(kivy_options['camera'])
providers.remove('opencv')
kivy_options['camera'] = tuple(providers)

import numpy as np
from camera4kivy import Preview
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.carousel import Carousel
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.utils import platform
from kivy.vector import Vector

from detector import detect
from solver import solve
from app import ui
from app import androidperm


__version__ = '0.4.1'

DEBUG = os.getenv('DEBUG')
ROOT_DIRPATH = pathlib.Path(__file__).parent

lgr = logging


class BidlyApp(App):
    def build(self):
        self.bidly = Bidly()
        return self.bidly

    def on_start(self):
        self.dont_gc = androidperm.AndroidPermissions(self.start_app)

    def on_stop(self):
        self.bidly.deal_box.camera_square.camera.disconnect_camera()

    def start_app(self):
        self.dont_gc = None
        Clock.schedule_once(self.connect_camera)

    def connect_camera(self, dt):
        self.bidly.deal_box.camera_square.camera.connect_camera()


class Bidly(BoxLayout):
    deal_box: ObjectProperty(None)
    interaction_box: ObjectProperty(None)

    ONNX_MODEL_FILE = 'best.onnx.1056' if DEBUG else 'best1184.onnx'
    ONNX_MODEL_PATH = ROOT_DIRPATH/'detector'/ONNX_MODEL_FILE

    MIN_OBJS_DETECTED = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        yolo5 = detect.Yolo5Opencv(detect.OpencvOnnxLoader())
        yolo5.load(self.ONNX_MODEL_PATH)
        self._model = yolo5
        lgr.debug("Loaded model from: %s", self.ONNX_MODEL_PATH)

    def detect_solve(self):
        pp = ui.popup("Detecting & Solving", "This could take a minute..")
        Clock.schedule_once(lambda dt: self._detect_solve())  # so that popup is instantly shown
        pp.dismiss()
        Clock.schedule_once(lambda dt: self.deal_box.load_next())

    def _detect_solve(self):
        lgr.info("Taking photo..")
        img_src = self.deal_box.camera_square.camera.capture()

        lgr.info("Handling image..")
        try:
            image_input = self._handle_image(img_src)
        except Exception as e:
            lgr.exception("Image handler failure")
            pp = ui.popup("Image handler failure", msg=repr(e), close_btn=True)
            pp.bind(on_dismiss=lambda _: self.restart())
            return

        lgr.info("Detecting cards..")
        detection = self._model.detect(image_input)
        if len(detection) < self.MIN_OBJS_DETECTED:
            ui.popup("Too few cards", msg="Too few cards detected; please retry", close_btn=True)
            return

        lgr.info("Solving deal..")
        try:
            solution = self._solve(detection)
        except Exception as e:
            lgr.exception("Solver failure")
            ui.popup("Solver failure", msg=repr(e), close_btn=True)
            return

        lgr.info("Displaying solution..")
        self.display(solution)

    def display(self, solution):
        hand, table = solution

        hand_label = BgcolorLabel()
        hand_label.display(hand)
        self.deal_box.add_widget(hand_label)

        n_buttons = len(self.interaction_box.children)  # so results are above buttons
        table_label = BgcolorLabel(font_size='21dp')
        table_label.display(table)
        self.interaction_box.add_widget(table_label, index=n_buttons)

    def restart(self):
        self.deal_box.restart()
        self.interaction_box.restart()

    def _handle_image(self, img_src) -> detect.ImageInput:
        image_handler = detect.get_image_handler(image_reader=detect.FsImageReader())
        image_handler.read(img_src)
        image_handler.validate()
        image_input = image_handler.preprocess()
        return image_input

    def _solve(self, detection: detect.CardDetection):
        solver = solve.BridgeSolver(detection, presenter=solve.MonoStringPresenter())
        missing, fp = solver.transform()
        if missing:
            raise ValueError(f"Missing cards: {', '.join(ui.display_name(n) for n in missing)}")

        solver.assign()
        solver.solve()
        return solver.present()


class DealBox(Carousel):
    camera_square = ObjectProperty(None)

    def restart(self):
        if len(self.slides) == 2:
            self.load_previous()
            Clock.schedule_once(lambda dt: self.remove_widget(self.slides[-1]), 0.5)

        self.camera_square.camera.play = True


class InteractionBox(BoxLayout):
    camera_button: ObjectProperty(None)
    restart_button: ObjectProperty(None)

    def restart(self):
        for widget in self.children[:]:
            if not isinstance(widget, Button):
                self.remove_widget(widget)

        self.camera_button.disabled = False
        self.restart_button.disabled = True


class CameraView(Camera):

    def capture(self):
        img = self.export_as_image()

        pixels = img.texture.pixels
        size = img.texture.size
        img_data = np.frombuffer(pixels, np.uint8).reshape((size[1], size[0], 4))

        lgr.debug("Capture image from camera, with shape: %s", img_data.shape)
        return img_data


class C4KCameraView(Preview):
    CAPTURE_SUBDIR = 'temp'
    CAPTURE_NAME = 'captured.jpg'

    def capture(self):
        if DEBUG:
            location = 'private'
        elif platform == 'android':
            from android.storage import app_storage_path
            location = f'{app_storage_path()}/DCIM'
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        capture_path = pathlib.Path(location)/self.CAPTURE_SUBDIR/self.CAPTURE_NAME

        if capture_path.exists():
            capture_path.unlink()

        self.capture_photo(location='private', subdir=self.CAPTURE_SUBDIR, name=self.CAPTURE_NAME)

        return capture_path


class BackgroundColor(Widget):
    pass


class BgcolorLabel(Label, BackgroundColor):
    label_text = StringProperty("")

    def display(self, text):
        self.label_text = text


### Reference only ###
class PongGame(Widget):
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)

    def on_touch_move(self, touch):  ##
        if touch.x < self.width/3:
            self.player1.center_y = touch.y
        if touch.x > self.width - self.width/3:
            self.player2.center_y = touch.y


class PongPaddle(Widget):
    def bounce_ball(self, ball):
        if self.collide_widget(ball):  ## or collide_point
            speedup  = 1.1
            offset = 0.02 * Vector(0, ball.center_y-self.center_y)
            ball.velocity =  speedup * (offset - ball.velocity)


if __name__ == '__main__':
    BidlyApp().run()
