import pathlib

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.vector import Vector

import detect
import solve


__version__ = '0.1.0'

FILEPATH = pathlib.Path(__file__)


class BidlyApp(App):
    def build(self):
        image_handler = detect.get_image_handler()
        image_handler.read(FILEPATH.parent/'deal5-md-sq.jpg')
        image_handler.validate()
        image_input = image_handler.preprocess()

        ONNX_MODEL_PATH = FILEPATH.parent.parent/'detector/yolov5/runs/train/r3/weights/best.onnx'
        yolo5 = detect.Yolo5Opencv(detect.OpencvOnnxLoader())
        yolo5.load(ONNX_MODEL_PATH)
        detection = yolo5.detect(image_input)

        solver = solve.BridgeSolver(detection, presenter=solve.StringPresenter())
        solver.transform()
        solver.assign()
        solver.solve()
        hand, table = solver.present()

        solution = Solution()
        solution.view(hand=hand, table=table)
        return solution


class Solution(GridLayout):
    hand = StringProperty(None)
    table = StringProperty(None)

    def view(self, hand, table):
        self.hand = hand
        self.table = table


class PongGame(Widget):
    ball = ObjectProperty(None)  #
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

    def on_touch_move(self, touch):
        if touch.x < self.width/3:
            self.player1.center_y = touch.y
        if touch.x > self.width - self.width/3:
            self.player2.center_y = touch.y

class PongBall(Widget):
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self):
        self.pos = Vector(*self.velocity) + self.pos


class PongPaddle(Widget):
    score = NumericProperty(0)

    def bounce_ball(self, ball):
        if self.collide_widget(ball):  #
            speedup  = 1.1
            offset = 0.02 * Vector(0, ball.center_y-self.center_y)
            ball.velocity =  speedup * (offset - ball.velocity)


if __name__ == '__main__':
    BidlyApp().run()
