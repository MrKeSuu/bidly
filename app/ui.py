"""Some UI helpers"""
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout


class PopupLabel(Label):
    pass


def popup(title, msg, close_btn=False, auto_dismiss=False):
    content = BoxLayout(orientation='vertical')

    label = PopupLabel(text=msg, valign='center', halign='center')
    content.add_widget(label)
    if close_btn:
        button = Button(text='Close', size_hint=(1, None))
        content.add_widget(button)

    popup = Popup(
        title=title, content=content,
        size_hint=(.8, .3), auto_dismiss=auto_dismiss)
    popup.open()

    if close_btn:
        button.bind(on_release=popup.dismiss)

    return popup


def display_name(name):
    rank, suit = name[:-1], name[-1]
    return f"{suit.upper()}{rank}"
