"""Some UI helpers"""
import random
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout

SUIT_UNICODE_MAP = {
    's': '\u2660',
    'h': '\u2661',
    'd': '\u2662',
    'c': '\u2663',
}


class PopupLabel(Label):
    pass


def popup(title, msg, close_btn=False, auto_dismiss=False):
    content = BoxLayout(orientation='vertical')

    label = PopupLabel(text=str(msg), valign='center', halign='center')
    content.add_widget(label)
    if close_btn:
        button = Button(text='Close', size_hint=(1, None))
        content.add_widget(button)

    popup = Popup(
        title=title.title(), content=content,
        size_hint=(.8, .3), auto_dismiss=auto_dismiss)
    popup.open()

    if close_btn:
        button.bind(on_release=popup.dismiss)

    return popup


def display_name(name, unicode=False):
    rank, suit = name[:-1], name[-1]
    suit = SUIT_UNICODE_MAP[suit] if unicode else suit.upper()
    return f"{suit}{rank}"


def random_tip():
    return random.choice([
        "Move phone closer to cards while ensure capturing all card symbols.",
        "When detecting, bidly only looks for the symbols in the corners of cards.",
    ])