import logging
import os
import pathlib
import random
import time

from camera4kivy import Preview
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen, RiseInTransition
from kivy.utils import platform

from detector import detect
from app import const, ui

DEBUG = os.getenv('DEBUG')
DETECTOR_DIRPATH = pathlib.Path(detect.__file__).parent

lgr = logging


LAYOUT = """
#:kivy 2.2.1
#:import ui app.ui

<MainScreen>:
    orientation: 'vertical'
    padding: (0, 0, 0, 10)
    spacing: 5

    camera_square: camera_square
    button_box: button_box

    on_enter:
        camera_square.camera.connect(); \
        button_box.camera_button.disabled = False; \

    on_pre_leave:
        camera_square.camera.disconnect_camera()

    CameraSquare:
        id: camera_square

    ButtonBox:
        id: button_box

        camera_button: camera_button

        Button:
            id: camera_button
            text: 'Detect & Solve'
            size_hint_y: None
            height: '48dp'
            disabled: True
            on_release:
                self.disabled = True; \
                root.detect_solve(); \

<ButtonBox>:
    pos_hint: {'bottom': 0}
    orientation: 'vertical'
    spacing: 5


<CameraSquare@FloatLayout>:
    size_hint: 1, None
    height: self.width / 3 * 4  # camera aspect ratio
    pos_hint: {'top': 1}

    camera: camera

    C4KCameraView:
        id: camera
        pos: root.pos

    BoxLayout:
        orientation: 'vertical'
        pos: root.pos

        BgcolorLabel:
            size: (camera.width, camera.height / 8)  # ensure square frame
            background_color: 0, 0, 0, 0.75

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
                # Player division guide lines
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
            background_color: 0, 0, 0, 0.75

<C4KCameraView>:
    aspect_ratio: '4:3'


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


class MainScreen(BoxLayout, Screen):
    camera_square: ObjectProperty(None)
    button_box: ObjectProperty(None)

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
        pp = ui.popup("Detecting", f"Tip: {self._random_tip()}")
        Clock.schedule_once(lambda dt: self._detect_solve())  # so that popup is instantly shown
        pp.dismiss()

    def restart(self):
        if not self.camera_square.camera.camera_connected:
            self.camera_square.camera.connect()

        self.button_box.restart()

    def _random_tip(self):
        return random.choice([
            "Move phone closer to cards while ensure capturing all card symbols.",
        ])

    def _detect_solve(self):
        try:
            img_src = self._capture()
        except Exception:
            self.restart()
            return

        try:
            detection = self._detect(img_src)
        except Exception:
            self.restart()
            return

        result_screen = self._get_result_screen()

        try:
            result_screen.display_image(img_src)
            self.manager.switch_to(result_screen, transition=RiseInTransition())
        except Exception as e:
            lgr.exception("Image loading failure")
            ui.popup("Image loading failure", msg=e, close_btn=True)
            self.restart()

        pp = ui.popup("Solving", "This could take a minute..")
        Clock.schedule_once(lambda dt: result_screen.process_detection(detection))
        pp.dismiss()

    def _capture(self):
        lgr.info("Capturing photo..")
        try:
            img_src = self.camera_square.camera.capture()
        except Exception as e:
            lgr.exception("Camera failure")
            ui.popup("Camera failure", msg=repr(e), close_btn=True)
            raise

        return img_src

    def _detect(self, img_src):
        lgr.info("Handling image..")
        try:
            image_input = self._handle_image(img_src)
        except Exception as e:
            lgr.exception("Image handler failure")
            ui.popup("Image handler failure", msg=repr(e), close_btn=True)
            raise

        lgr.info("Detecting cards..")
        detection = self._model.detect(image_input)
        if len(detection) < self.MIN_OBJS_DETECTED:
            ui.popup("Too few cards", msg="Too few cards detected; please retry", close_btn=True)
            raise ValueError("Too few cards")

        return detection

    def _get_result_screen(self):
        result_screen = self.manager.get_screen(const.RESULT_SCREEN)
        return result_screen

    def _handle_image(self, img_src) -> detect.ImageInput:
        image_handler = detect.get_image_handler(image_reader=detect.FsExifImageReader())
        image_handler.read(img_src)
        image_handler.validate()
        image_input = image_handler.preprocess()
        return image_input


class ButtonBox(BoxLayout):
    camera_button: ObjectProperty(None)

    def restart(self):
        self.camera_button.disabled = False


class C4KCameraView(Preview):
    CAPTURE_SUBDIR = 'temp'
    CAPTURE_NAME = 'captured.jpg'

    def connect(self):
        self.connect_camera(sensor_resolution=(1600, 1200))

    def capture(self):
        capture_path = self.get_capture_path()

        if capture_path.exists():
            capture_path.unlink()

        self.capture_photo(location='private', subdir=self.CAPTURE_SUBDIR, name=self.CAPTURE_NAME)

        # Ugly, but checking filesize to see if capture_photo completed which is on another thread
        filesize = self._filesize(capture_path)
        while filesize == 0 or filesize < self._filesize(capture_path):
            lgr.debug("Waiting for capture done..")
            filesize = self._filesize(capture_path)
            time.sleep(0.05)
        self.disconnect_camera()
        lgr.debug("Capture completed")

        return capture_path

    def get_capture_path(self):
        if DEBUG:
            capture_location = 'private'
        elif platform == 'android':
            from android.storage import app_storage_path
            capture_location = f'{app_storage_path()}/DCIM'
        else:
            raise ValueError(f"Unsupported platform: {platform}")

        capture_path = pathlib.Path(capture_location)/self.CAPTURE_SUBDIR/self.CAPTURE_NAME
        return capture_path

    @staticmethod
    def _filesize(path):
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
