import logging
import os

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.carousel import Carousel
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, FallOutTransition
from kivy.uix.widget import Widget
from kivy.utils import platform
from kivy.vector import Vector

from app import const, ui
from detector import detect
from solver import solve

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

    problematic_cards: []

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

<UserAssignment>:
    pos_hint:{'top': 1}
    anim_move_duration: 0.2

<AssignmentSlide>:
    card: ''
    n_remaining: 1

    Button:
        text: ''
        disabled: True
    Button:
        text: 'N'
        on_release: root.assign_to_player('north')
    Button:
        text: ''
        disabled: True

    Button:
        text: 'W'
        on_release: root.assign_to_player('west')
    BgcolorLabel:
        text: ui.display_name(root.card, unicode=True)+'?' if root.card else ''
    Button:
        text: 'E'
        on_release: root.assign_to_player('east')

    Button:
        text: ''
        disabled: True
    Button:
        text: 'S'
        on_release: root.assign_to_player('south')
    Button:
        text: str(root.n_remaining)+' left to assign'
        disabled: True
        color: 0.9, 0.9, 0.9, 1

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

    solver: ObjectProperty(None)
    problematic_cards: ListProperty([])

    def __init__(self, **kwargs):
        Builder.load_string(LAYOUT)
        super().__init__(**kwargs)

        self.bind(problematic_cards=self.on_problematic_cards)

    def process_detection(self, detection: detect.CardDetection):
        self.solver = solve.BridgeSolver(detection, presenter=solve.MonoStringPresenter())

        try:
            transf_results = self._transform_detection()
            if transf_results.missings:
                self.show_assignment_widget(transf_results.missings)
                return

            assign_results = self._assign_detection(transf_results.cards)

            solution = self._solve(assign_results.cards)

        except Exception as e:
            lgr.exception("Solver failure")
            ui.popup("Solver failure", msg=e, close_btn=True)
            return

        self.display_solution(solution)

    def display_image(self, img_path):
        ImageWidget = AndroidAsyncImage if platform == 'android' else AsyncImage
        captured_image = ImageWidget(source=str(img_path), fit_mode='cover', nocache=True)
        captured_image.reload()  # to ensure no cache
        self.deal_box.add_widget(captured_image)

    def display_solution(self, solution):
        lgr.info("Displaying solution..")
        hand, table = solution

        hand_label = AdaptiveBgcolorLabel()
        hand_label.display(hand)
        self.deal_box.add_widget(hand_label)

        table_label = BgcolorLabel()
        table_label.display(table)
        n_buttons = len(self.interaction_box.children)  # so results are above buttons
        self.interaction_box.add_widget(table_label, index=n_buttons)

        self.deal_box.load_next()

    def show_assignment_widget(self, problematic_cards):
        self.problematic_cards = problematic_cards

    def on_enter(self, *args):
        if len(self.deal_box.slides) > 1:
            Clock.schedule_once(lambda dt: self.deal_box.load_next(), 0.2)

        self.interaction_box.restart_button.disabled = False

    def on_problematic_cards(self, instance, value):
        # for restart
        if not value and not instance.interaction_box.has_assignment_widget():
            return

        # for init
        if not instance.interaction_box.has_assignment_widget():
            instance.interaction_box.add_assignment_widget(value)
            return

        # no problematic cards left
        if not value:
            instance.interaction_box.remove_assignment_widget()
            pp = ui.popup("Solving", "This could take a minute..")
            transformed_cards = self.deal_box.detection_data
            Clock.schedule_once(lambda dt: self._run_assign_downstream(transformed_cards))
            pp.dismiss()
            return

        # still some problematic cards remained to assign
        instance.interaction_box.assignment_widget.load_next()

    def restart(self):
        self.deal_box.restart()
        self.interaction_box.restart()

        self.problematic_cards = []

        main_screen = self.manager.get_screen(const.MAIN_SCREEN)
        self.manager.switch_to(main_screen, transition=FallOutTransition())

    def _transform_detection(self) -> solve.TransformationResults:
        transf_results = self.solver.transform()

        self.deal_box.detection_data = transf_results.cards

        return transf_results

    def _run_assign_downstream(self, transformed_cards):
        try:
            assign_results = self._assign_detection(transformed_cards)

            solution = self._solve(assign_results.cards)

        except Exception as e:
            lgr.exception("Solver failure")
            ui.popup("Solver failure", msg=e, close_btn=True)
            return

        self.display_solution(solution)

    def _assign_detection(self, transformed_cards) -> solve.AssignmentResults:
        lgr.info("Assigning cards..")
        assign_results = self.solver.assign(transformed_cards)

        if assign_results.not_assigned:
            # TODO let user assign
            raise ValueError(f"Unassigned cards: {', '.join(ui.display_name(n) for n in assign_results.not_assigned)}")

        return assign_results

    def _solve(self, assigned_cards):
        lgr.info("Solving deal..")
        self.solver.solve(assigned_cards)

        solution = self.solver.present()
        return solution


class DealBox(Carousel):
    detection_data: ListProperty([])

    def restart(self):
        self.clear_widgets()


class InteractionBox(BoxLayout):
    restart_button: ObjectProperty(None)

    def restart(self):
        for widget in self.children[:]:
            if not isinstance(widget, Button):
                self.remove_widget(widget)

        self.restart_button.disabled = True

    def has_assignment_widget(self):
        for widget in self.children:
            if isinstance(widget, UserAssignment):
                return True
        return False

    def add_assignment_widget(self, problematic_cards):
        assignment_widget = UserAssignment()
        assignment_widget.prepare_slides(problematic_cards)

        n_buttons = len(self.children)  # so widget is above buttons
        self.add_widget(assignment_widget, index=n_buttons)
        self.assignment_widget = assignment_widget

    def remove_assignment_widget(self):
        for widget in self.children[:]:
            if isinstance(widget, UserAssignment):
                self.remove_widget(widget)
        self.assignment_widget = None


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
        super().display(text)
        self.text_width = len(text.split('\n')[1])


class UserAssignment(Carousel):

    def on_touch_move(self, touch):
        return  # disable change of slides by swiping 

    def prepare_slides(self, problematic_cards):
        for card in problematic_cards[:]:
            n_remaining = len(problematic_cards) - problematic_cards.index(card)
            self._add_assignment_slide(card, n_remaining)

    def assign_card(self, card, player):
        player_coord_x, player_coord_y = self._get_player_coords(player)
        detection_record = dict(
            name=card,
            center_x=player_coord_x,
            center_y=player_coord_y,
            confid=1,
        )
        self.parent.parent.deal_box.detection_data.append(detection_record)

        self.parent.parent.problematic_cards.remove(card)

    def _add_assignment_slide(self, card, n_remaining):
        slide = AssignmentSlide(cols=3)
        slide.card = card
        slide.n_remaining = n_remaining
        self.add_widget(slide)

    @staticmethod
    def _get_player_coords(player):
        # Note: origin is at top left corner, instead of bottom left
        if player == solve.converter.HAND_N:
            return (0.5, 0.25)
        if player == solve.converter.HAND_S:
            return (0.5, 0.75)
        if player == solve.converter.HAND_W:
            return (0.25, 0.5)
        if player == solve.converter.HAND_E:
            return (0.75, 0.5)

        raise ValueError(f"Bad 'player' provided: {player}")


class AssignmentSlide(GridLayout):
    card: StringProperty("")
    n_remaining: NumericProperty(1)

    def assign_to_player(self, player):
        # to account for an auto-added relativelayout
        self.parent.parent.assign_card(self.card, player)


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
