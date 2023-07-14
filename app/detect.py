import abc
import dataclasses
import logging
import typing

import cv2
import numpy as np


IMAGE_WIDTH = 1184
IMGAE_HEIGHT = 1184
MIN_CONFIDENCE = 0.3

lgr = logging


# Abstract #

@dataclasses.dataclass
class ImageInput:
    data: np.ndarray


CardDetection = typing.List[typing.Dict]  # detection records


class IYoloLoader(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def load(src):
        pass


class YoloModel(abc.ABC):
    loader: IYoloLoader

    def __init__(self, loader):
        self.loader = loader

    def load(self, src):
        self._model = self.loader.load(src)

    @abc.abstractmethod
    def detect(self, image: ImageInput) -> CardDetection:
        pass


class IImageReader(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def read(src):
        pass


class IImageValidator(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def validate(cls, image):
        pass


class IImagePreprocessor(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def preprocess(cls, image):
        pass


class ImageHandlerBase(abc.ABC):
    reader: IImageReader
    validators: typing.List[IImageValidator]
    preprocessors: typing.List[IImagePreprocessor]

    def __init__(self, reader, validators, preprocesssors):
        self.reader = reader
        self.validators = validators
        self.preprocessors = preprocesssors

    def read(self, src):
        self._image = self.reader.read(src)

    def validate(self):
        for validator in self.validators:
            validator.validate(self._image)  # may raise ValueError

    @abc.abstractmethod
    def preprocess(self) -> ImageInput:
        pass
