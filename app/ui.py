"""Some UI helpers"""
from kivy.uix.label import Label
from kivy.uix.popup import Popup


def show_msg(title, msg):
    content = Label(text=msg)
    popup = Popup(title=title, content=content, size_hint=(0.9, None), height=400)
    popup.open()

    return popup


def display_name(name):
    rank, suit = name[:-1], name[-1]
    return f"{suit.upper()}{rank}"
