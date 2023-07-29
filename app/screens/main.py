import logging
import os
import pathlib
import time

import numpy as np
from camera4kivy import Preview
from kivy.clock import Clock
from kivy.lang import Builder
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

DEBUG = os.getenv('DEBUG')
DETECTOR_DIRPATH = pathlib.Path(detect.__file__).parent

lgr = logging


LAYOUT = """
#:kivy 2.2.1
#:import ui app.ui

<Bidly>:
    orientation: 'vertical'
    padding: (0, 0, 0, 10)
    spacing: 5

    deal_box: deal_box
    interaction_box: interaction_box

    DealBox:
        id: deal_box

    InteractionBox:
        id: interaction_box

        camera_button: camera_button
        restart_button: restart_button

        Button:
            id: camera_button
            text: 'Detect & Solve'
            size_hint_y: None
            height: '48dp'
            on_release:
                deal_box.camera_square.camera.play = False; self.disabled = True; \
                root.detect_solve(); restart_button.disabled = False

        Button:
            id: restart_button
            text: 'Restart'
            size_hint_y: None
            height: '48dp'
            disabled: True
            on_release: root.restart()


<DealBox>:
    size_hint: 1, None
    size: self.width, self.width / 3 * 4
    anim_move_duration: 0.3

    camera_square: camera_square

    CameraSquare:
        id: camera_square

<InteractionBox>:
    orientation: 'vertical'
    spacing: 5


<CameraSquare@FloatLayout>:
    camera: camera

    C4KCameraView:
        id: camera
        pos: root.pos

    BoxLayout:
        orientation: 'vertical'

        BgcolorLabel:
            size: (camera.width, camera.height / 8)  # ensure square frame
            background_color: 0, 0, 0, 0.5

        Label:
            pos: camera.pos
            size_hint_y: None
            height: camera.width

            text: 'N'
            color: 1, 1, 1, 0.5
            font_size: '50dp'
            text_size: self.size
            valign: 'top'
            halign: 'center'

            canvas:
                Color:
                    rgba: 1, 1, 1, 1
                Line:
                    points: 0, self.center_y + self.height/2, self.width, self.center_y - self.height/2
                    dash_length: 10
                    dash_offset: 10
                Line:
                    points: 0, self.center_y - self.height/2, self.width, self.center_y + self.height/2
                    dash_length: 10
                    dash_offset: 10

        BgcolorLabel:
            size: (camera.width, camera.height / 8)  # ensure square frame
            background_color: 0, 0, 0, 0.5

<CameraView>:
    resolution: (1600, 1200)  # one of the supported
    play: True
    fit_mode: 'cover'

    canvas.before:
        PushMatrix
        Rotate:
            angle: -90
            origin: self.center
    canvas.after:
        PopMatrix

<C4KCameraView>:
    aspect_ratio: '4:3'

<BgcolorLabel>:
    text: str(self.label_text)

    font_size: '18dp'
    font_name: 'app/fonts/FreeMono.ttf'
    color: 0.15, 0.15, 0.15, 1  # from medium.com

    text_size: self.size
    valign: 'center'
    halign: 'center'


<PopupLabel>:
    # https://stackoverflow.com/questions/66018633/is-there-a-way-to-adjust-the-size-of-content-in-a-kivy-popup
    text_size: self.size

# Define your background color Template
<BackgroundColor@Widget>
    background_color: 1, 1, 1, 1
    canvas.before:
        Color:
            rgba: root.background_color
        Rectangle:
            size: self.size
            pos: self.pos

# Now you can simply Mix the `BackgroundColor` class with almost
# any other widget... to give it a background.
<BgcolorLabel@Label+BackgroundColor>
    # Default the background color for this label
    # to r 0, g 0, b 0, a 0
    background_color: 1, 1, 1, 1
"""


class Bidly(BoxLayout):
    deal_box: ObjectProperty(None)
    interaction_box: ObjectProperty(None)

    ONNX_MODEL_FILE = 'best.onnx.1056' if DEBUG else 'best1184.onnx'
    ONNX_MODEL_PATH = DETECTOR_DIRPATH/ONNX_MODEL_FILE

    MIN_OBJS_DETECTED = 50

    def __init__(self, **kwargs):
        Builder.load_string(LAYOUT)
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
        lgr.info("Capturing photo..")
        try:
            img_src = self.deal_box.camera_square.camera.capture()
        except Exception:
            lgr.exception("Camera failure")
            pp = ui.popup("Camera failure", msg=repr(e), close_btn=True)
            pp.bind(on_dismiss=lambda _: self.restart())
            return

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

        table_label = BgcolorLabel(font_size='21dp')
        table_label.display(table)
        n_buttons = len(self.interaction_box.children)  # so results are above buttons
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
            capture_location = 'private'
        elif platform == 'android':
            from android.storage import app_storage_path
            capture_location = f'{app_storage_path()}/DCIM'
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        capture_path = pathlib.Path(capture_location)/self.CAPTURE_SUBDIR/self.CAPTURE_NAME

        if capture_path.exists():
            capture_path.unlink()

        self.capture_photo(location='private', subdir=self.CAPTURE_SUBDIR, name=self.CAPTURE_NAME)

        # Ugly, but checking filesize to see if capture_photo is done on another thread
        filesize = self._filesize(capture_path)
        while filesize == 0 or filesize < self._filesize(capture_path):
            filesize =self._filesize(capture_path)
            time.sleep(0.1)
        lgr.debug("Capture completed")

        return capture_path

    @staticmethod
    def _filesize(path):
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0


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
