"""Detector interfaces/Classes used by app."""
import abc
import dataclasses
import logging
import os
import typing

import cv2
import numpy as np

DEBUG = os.getenv('DEBUG')
IMAGE_WIDTH = 1056 if DEBUG else 1184
IMAGE_HEIGHT = 1056 if DEBUG else 1184
MIN_CONFIDENCE = 0.9

lgr = logging


# Abstract #

CardDetection = typing.List[typing.Dict]  # detection records


@dataclasses.dataclass
class ImageInput:
    data: np.ndarray


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
        self._model = self.loader.load(str(src))

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
        lgr.info("Read image with shape: %s", self._image.shape)

    def validate(self):
        for validator in self.validators:
            validator.validate(self._image)  # may raise ValueError

    @abc.abstractmethod
    def preprocess(self) -> ImageInput:
        pass


# Impl #

class ImageHandler(ImageHandlerBase):
    def preprocess(self) -> ImageInput:
        image = self._image
        for preprocessor in self.preprocessors:
            image = preprocessor.preprocess(image)

        image_input = ImageInput(data=image)
        return image_input


class Yolo5Opencv(YoloModel):
    MIN_CLASS_SCORE = 0.5
    MIN_NMS = 0.45

    def detect(self, image: ImageInput) -> CardDetection:
        image_blob = cv2.dnn.blobFromImage(
            image.data,
            scalefactor=1/255,
            size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            mean=[0,0,0],
            swapRB=True,
            crop=False,
        )
        lgr.debug("Got blob with shape: %s", image_blob.shape)
        self._model.setInput(image_blob)
        outputs = self._model.forward(self._model.getUnconnectedOutLayersNames())

        detection = self._postprocess(outputs)
        lgr.debug("Detection with min confid: %s",
                  min(detection, key=lambda d: d['confidence'], default=None))

        return detection

    def _postprocess(self, outputs):
        # Lists to hold respective values while unwrapping
        class_ids = []
        confidences = []
        boxes = []

        rows = outputs[0].shape[1]

        # Iterate through detections
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= MIN_CONFIDENCE:
                    classes_scores = row[5:]
                    # Get the index of max class score.
                    class_id = np.argmax(classes_scores)
                    #  Continue if the class score is above threshold.
                    if (classes_scores[class_id] > self.MIN_CLASS_SCORE):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = cx - w/2
                        top = cy - h/2
                        width = w
                        height = h
                        box = np.array([left, top, width, height]) / IMAGE_WIDTH
                        boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, self.MIN_NMS)

        records = []
        for i in indices:
            x_, y_, w, h = boxes[i]
            record = {
                'x': x_ + w/2,
                'y': y_ + h/2,
                'w': w,
                'h': h,
                'confidence': confidences[i],
                'class_id': class_ids[i]
            }
            records.append(record)

        return records


class OpencvOnnxLoader(IYoloLoader):

    @staticmethod
    def load(src):
        model = cv2.dnn.readNetFromONNX(src)
        return model


class FsImageReader(IImageReader):
    """Read image from a file system path"""

    @staticmethod
    def read(src):
        path = str(src)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"Can't load image at: {path}")

        return image


class FsExifImageReader(IImageReader):
    """Read image from file system considering EXIF info."""

    @staticmethod
    def read(src):
        path = str(src)
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Can't load image at: {path}")

        return image


class BgraReader(IImageReader):

    @staticmethod
    def read(src):
        image = cv2.cvtColor(src, cv2.COLOR_BGRA2RGB)
        return image


class MinSizeValidator(IImageValidator):
    MIN_WIDTH = IMAGE_WIDTH
    MIN_HEIGHT = IMAGE_HEIGHT

    def validate(cls, image):
        height, width = image.shape[0], image.shape[1]
        if height < cls.MIN_WIDTH or width < cls.MIN_HEIGHT:
            raise ValueError(f"Image resolution {width}x{height} too low")


class ImageCrop(IImagePreprocessor):
    """Crop image to 1:1"""

    def preprocess(cls, image):
        height, width = image.shape[0], image.shape[1]

        # Make dimensions even for easier centered cropping
        if height % 2 != 0:
            image = image[:-1, :]
            height -= 1
        if width % 2 != 0:
            image = image[:, :-1]
            width -= 1

        margin = abs(height - width) // 2
        if height > width:
            cropped = image[margin:-margin, :]
        elif height < width:
            cropped = image[:, margin:-margin]
        else:
            cropped = image

        assert cropped.shape[0] == cropped.shape[1], "1:1 cropped image is not sqaure"

        return cropped


class ImageResize(IImagePreprocessor):
    TARGET_SIZE = (IMAGE_WIDTH ,IMAGE_HEIGHT)

    def preprocess(cls, image):
        return cv2.resize(image, cls.TARGET_SIZE)


def get_image_handler(image_reader=FsImageReader()):
    min_size = MinSizeValidator()
    resize = ImageResize()
    crop = ImageCrop()
    image_handler = ImageHandler(
        image_reader,
        validators=[min_size],
        preprocesssors=[resize, crop])
    return image_handler
