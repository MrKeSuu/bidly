"""Some UI helpers"""
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout


def show_msg(title, msg, auto_dismiss=False, add_button=False):
    content = BoxLayout(orientation='vertical')

    label = Label(text=msg, text_size=(800, None))
    label.bind(size=lambda ins: ins.texture_size)
    content.add_widget(Label(text=msg))

    if add_button:
        button = Button(text='Close')
        content.add_widget(button)

    popup = Popup(
        title=title, content=content,
        size_hint=(0.9, None), height=400, auto_dismiss=auto_dismiss)
    popup.open()

    if add_button:
        button.bind(on_release=popup.dismiss)

    return popup


def display_name(name):
    rank, suit = name[:-1], name[-1]
    return f"{suit.upper()}{rank}"
