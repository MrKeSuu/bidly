import logging
import pathlib

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.vector import Vector

from detector import detect
from solver import solve
from app import ui


__version__ = '0.3.2'

ROOT_DIRPATH = pathlib.Path(__file__).parent

lgr = logging


class BidlyApp(App):
    def build(self):
        bidly = Bidly()
        return bidly


class Bidly(BoxLayout):
    deal_box: ObjectProperty(None)
    interaction_box: ObjectProperty(None)

    ONNX_MODEL_PATH = ROOT_DIRPATH/'detector/best1056.onnx'

    MIN_OBJS_DETECTED = 50

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        yolo5 = detect.Yolo5Opencv(detect.OpencvOnnxLoader())
        yolo5.load(self.ONNX_MODEL_PATH)
        self._model = yolo5
        lgr.debug("Loaded model from: %s", self.ONNX_MODEL_PATH)

    def detect_solve(self):
        pp = ui.popup("Detectin & Solving", "This could take a minute")
        Clock.schedule_once(lambda dt: self._detect_solve())  # so that popup is instantly shown
        pp.dismiss()

    def _detect_solve(self):
        lgr.info("Taking photo..")
        img_data = self.deal_box.camera.capture()

        lgr.info("Handling image..")
        try:
            image_input = self._handle_image(img_data)
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

        # self.interaction_box.clear_widgets()

        n_buttons = len(self.interaction_box.children)  # so results are above buttons

        hand_label = BgcolorLabel(font_size='9dp')
        hand_label.display(hand)
        self.interaction_box.add_widget(hand_label, index=n_buttons)

        table_label = BgcolorLabel(size_hint_y=0.7)
        table_label.display(table)
        self.interaction_box.add_widget(table_label, index=n_buttons)

    def restart(self):
        for widget in self.interaction_box.children[:]:
            if not isinstance(widget, Button):
                self.interaction_box.remove_widget(widget)

        self.deal_box.camera.play = True
        self.interaction_box.restart()

    def _handle_image(self, img_data) -> detect.ImageInput:
        image_handler = detect.get_image_handler(image_reader=detect.BgraReader())
        image_handler.read(img_data)
        image_handler.validate()
        image_input = image_handler.preprocess()
        return image_input

    def _solve(self, detection: detect.CardDetection):
        solver = solve.BridgeSolver(detection, presenter=solve.StringPresenter())
        missing, fp = solver.transform()
        if missing:
            raise ValueError(f"Missing cards: {', '.join(ui.display_name(n) for n in missing)}")

        solver.assign()
        solver.solve()
        return solver.present()


class DealBox(FloatLayout):
    camera = ObjectProperty(None)


class InteractionBox(BoxLayout):
    camera_button: ObjectProperty(None)
    restart_button: ObjectProperty(None)

    def restart(self):
        self.camera_button.disabled = False
        self.restart_button.disabled = True


class CameraWidget(Camera):

    def capture(self):
        img = self.export_as_image()

        pixels = img.texture.pixels
        size = img.texture.size
        img_data = np.frombuffer(pixels, np.uint8).reshape((size[1], size[0], 4))

        lgr.debug("Capture image from camera, with shape: %s", img_data.shape)
        return img_data


class BackgroundColor(Widget):
    pass


class BgcolorLabel(Label, BackgroundColor):
    label_text = StringProperty("")

    def display(self, text):
        self.label_text = text


class PongGame(Widget):
    ball = ObjectProperty(None)
    player1 = ObjectProperty(None)
    player2 = ObjectProperty(None)

    def serve_ball(self, vel=(4, 0)):
        self.ball.center = self.center
        self.ball.velocity = vel

    def update(self, dt):
        self.ball.move()

        # bounce off paddles
        self.player1.bounce_ball(self.ball)
        self.player2.bounce_ball(self.ball)

        # bounce off top and bottom
        if (self.ball.y < 0) or (self.ball.top > self.height):
            self.ball.velocity_y *= -1

        # went off to a side to score point?
        if self.ball.x < self.x:
            self.player2.score += 1
            self.serve_ball(vel=(4, 0))
        if self.ball.right > self.width:
            self.player1.score += 1
            self.serve_ball(vel=(-4, 0))

    def on_touch_move(self, touch):  ##
        if touch.x < self.width/3:
            self.player1.center_y = touch.y
        if touch.x > self.width - self.width/3:
            self.player2.center_y = touch.y


class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):  ## or collide_point
            speedup  = 1.1
            offset = 0.02 * Vector(0, ball.center_y-self.center_y)
            ball.velocity =  speedup * (offset - ball.velocity)


if __name__ == '__main__':
    BidlyApp().run()
