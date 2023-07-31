import logging
import os

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.carousel import Carousel
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, FallOutTransition
from kivy.uix.widget import Widget
from kivy.utils import platform
from kivy.vector import Vector

from app import const, ui

DEBUG = os.getenv('DEBUG')

lgr = logging


LAYOUT = """
#:kivy 2.2.1
#:import ui app.ui

<ResultScreen>:
    orientation: 'vertical'
    padding: (0, 0, 0, 10)
    spacing: 5

    deal_box: deal_box
    interaction_box: interaction_box

    DealBox:
        id: deal_box

    InteractionBox:
        id: interaction_box

        restart_button: restart_button

        Button:
            id: restart_button
            text: 'Restart'
            size_hint_y: None
            height: '48dp'
            on_release: root.restart()


<PopupLabel>:
    # https://stackoverflow.com/questions/66018633/is-there-a-way-to-adjust-the-size-of-content-in-a-kivy-popup
    text_size: self.size


<DealBox>:
    size_hint: 1, None
    height: self.width
    pos_hint: {'top': 1}
    anim_move_duration: 0.2


<InteractionBox>:
    pos_hint: {'bottom': 0}
    orientation: 'vertical'
    spacing: 5


<AndroidAsyncImage>:
    # Android needs a rotation as kivy image does not read EXIF info.
    canvas.before:
        PushMatrix
        Rotate:
            angle: -90
            axis: 0, 0, 1
            origin: root.center
    canvas.after:
        PopMatrix

<AdaptiveBgcolorLabel>:
    font_size: str(380//self.text_width)+'sp'

<BgcolorLabel>:
    text: str(self.label_text)

    font_size: '25sp'
    font_name: 'app/fonts/FreeMono.ttf'
    color: 0.15, 0.15, 0.15, 1  # from medium.com

    text_size: self.size
    valign: 'center'
    halign: 'center'


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


class ResultScreen(BoxLayout, Screen):
    deal_box: ObjectProperty(None)
    interaction_box: ObjectProperty(None)

    def __init__(self, **kwargs):
        Builder.load_string(LAYOUT)
        super().__init__(**kwargs)

    def display(self, img_path, solution):
        hand, table = solution

        ImageWidget = AndroidAsyncImage if platform == 'android' else AsyncImage
        captured_image = ImageWidget(source=str(img_path), fit_mode='cover')
        self.deal_box.add_widget(captured_image)

        hand_label = AdaptiveBgcolorLabel()
        hand_label.display(hand)
        self.deal_box.add_widget(hand_label)

        table_label = BgcolorLabel()
        table_label.display(table)
        n_buttons = len(self.interaction_box.children)  # so results are above buttons
        self.interaction_box.add_widget(table_label, index=n_buttons)

    def on_enter(self, *args):
        if len(self.deal_box.slides) > 1:
            Clock.schedule_once(lambda dt: self.deal_box.load_next(), 0.2)

        self.interaction_box.restart_button.disabled = False

    def restart(self):
        self.deal_box.restart()
        self.interaction_box.restart()

        main_screen = self.manager.get_screen(const.MAIN_SCREEN)
        self.manager.switch_to(main_screen, transition=FallOutTransition())


class DealBox(Carousel):

    def restart(self):
        self.clear_widgets()


class InteractionBox(BoxLayout):
    restart_button: ObjectProperty(None)

    def restart(self):
        for widget in self.children[:]:
            if not isinstance(widget, Button):
                self.remove_widget(widget)

        self.restart_button.disabled = True


class AndroidAsyncImage(AsyncImage):
    pass


class BackgroundColor(Widget):
    pass


class BgcolorLabel(Label, BackgroundColor):
    label_text = StringProperty("")

    def display(self, text):
        self.label_text = text


class AdaptiveBgcolorLabel(BgcolorLabel):
    label_text = StringProperty("")
    text_width = NumericProperty(24)

    def display(self, text):
        print(text.split('\n'))
        super().display(text)
        self.text_width = len(text.split('\n')[0])


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