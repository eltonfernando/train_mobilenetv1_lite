# -*- coding: utf-8 -*-
import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
import torch


class Preprocessor:
    def __init__(self, input_size=640):
        self.input_size = input_size  # Define o tamanho da entrada do modelo (ex.: 640x640)

    def letterbox(self, image, boxes, input_size=640):
        """
        Redimensiona a imagem mantendo a proporção original, e preenche com padding para atingir o input_size.
        Ajusta também as bounding boxes.
        """
        height, width = image.shape[:2]
        scale = min(input_size / height, input_size / width)
        new_width, new_height = int(width * scale), int(height * scale)

        # Redimensionar a imagem com base na nova escala
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Adicionar padding (letterbox)
        pad_w = (input_size - new_width) // 2
        pad_h = (input_size - new_height) // 2
        padded_image = cv2.copyMakeBorder(resized_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        padded_image = cv2.resize(padded_image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        # Ajustar bounding boxes conforme o redimensionamento e padding
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w  # Ajustar coordenadas x (xmin, xmax)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h  # Ajustar coordenadas y (ymin, ymax)

        return padded_image, boxes

    def normalize(self, image):
        """
        Normaliza os valores de pixel para o intervalo [0, 1].
        """
        return image.astype(np.float32) / 255.0

    def to_chw(self, image):
        """
        Converte a imagem do formato HWC para CHW.
        """
        return np.transpose(image, (2, 0, 1))  # De (H, W, C) para (C, H, W)

    def preprocess(self, image, boxes):
        """
        Pipeline de pré-processamento completo.
        """
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        # Redimensiona a imagem com letterbox
        image, boxes = self.letterbox(image, boxes, self.input_size)
        # for x_min, y_min, x_max, y_max in boxes:
        #    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        # cv2.imwrite(f"letterbox{np.random.randint(0, 1000)}.jpg", image)

        # Normaliza os valores de pixel
        image = self.normalize(image)
        image = self.to_chw(image)

        return image, boxes


class VOCDataset:
    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.preprocessor = Preprocessor(input_size=224)

        # if the labels file exists, read in the class names

        self.__load_classe()

    def __load_classe(self):
        path = self.root / "labels.txt"
        if not os.path.isfile(path):
            raise FileNotFoundError("VOC Labels file not found: " + path)

        class_string = ""
        with open(path, "r") as infile:
            for line in infile:
                class_string += line.rstrip()

        classes = class_string.split(",")
        # prepend BACKGROUND as first class
        classes.insert(0, "BACKGROUND")
        classes = [elem.replace(" ", "") for elem in classes]
        self.class_names = tuple(classes)
        logging.info("VOC Labels read from file: " + str(self.class_names))

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)

        # image, boxes = self.preprocessor.preprocess(image, boxes)

        # image = torch.from_numpy(image, dtype=torch.float32)
        # logging.error("image: " + str(image.shape) + " boxes: " + str(boxes.shape))
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        if len(objects) == 0:
            return self._gen_random_background()
            raise Exception(f"xml sem box {annotation_file}")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find("name").text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find("bndbox")

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find("xmin").text) - 1
                y1 = float(bbox.find("ymin").text) - 1
                x2 = float(bbox.find("xmax").text) - 1
                y2 = float(bbox.find("ymax").text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find("difficult").text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        if len(boxes) == 0:
            return self._gen_random_background()
            raise Exception(f"box não pode ser nulo {annotation_file} {self.class_dict} {boxes}")
        return (np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(is_difficult, dtype=np.uint8))

    def _gen_random_background(self):
        return (np.array([[10, 30, 40, 50]], dtype=np.float32), np.array([0], dtype=np.int64), np.array([0], dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
