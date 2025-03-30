from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytesseract

if TYPE_CHECKING:
    from . import OCRPlugin

from PIL import Image, UnidentifiedImageError
from panoptic.core.task.task import Task
from panoptic.models import Instance, Property, ImageProperty, DbCommit

logger = logging.getLogger('PanopticOCR:OCRTask')


class ComputeOCRTask(Task):
    def __init__(self, plugin: OCRPlugin, instance: Instance, prop: Property, model: str):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.instance = instance
        self.name = f'Compute OCR'
        self.prop = prop
        self.model = model

    async def run(self):
        instance = self.instance
        try:
            image = self.preprocess_image(instance.url)
        except Exception as e:
            logger.error(f"Failed to preprocess image for instance {instance.id}: {e}")
            return

        try:
            ocr_function = self._get_ocr_function(self.model)
        except ValueError as e:
            logger.error(f"Invalid OCR model '{self.model}' provided: {e}")
            return

        try:
            text = await self._async(ocr_function, image)
        except Exception as e:
            logger.error(f"OCR processing failed for instance {instance.id}: {e}")
            return

        try:
            commit = DbCommit()
            commit.image_values.append(ImageProperty(
                property_id=self.prop.id,
                sha1=instance.sha1,
                value=text
            ))
            res = await self.project.do(commit)

            # Notify UI
            self.project.ui.commits.append(commit)
            self.project._project.db.on_import_instance.emit(commit.image_values[0])
            return res.image_values[0]
        except Exception as e:
            logger.error(f"Failed to commit OCR results for instance {instance.id}: {e}")
            return

    def make_ocr_doctr(self, image: Image.Image):
        try:
            image_arr = np.asarray(image)
            result = self.plugin._model([image_arr])
            res = ""
            js = result.export()
            for block in js.get('pages', [])[0].get('blocks', []):
                for line in block.get('lines', []):
                    for word in line.get('words', []):
                        if word.get('confidence', 0) > 0.30:
                            res += " " + word.get('value', '')
            return res.strip()
        except Exception as e:
            logger.error(f"doctr OCR failed: {e}")
            return ""

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise
        except UnidentifiedImageError:
            logger.error(f"Corrupted or unreadable image: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while opening image: {e}")
            raise

        params = self.plugin.params
        x, y, w, h = params.crop_x, params.crop_y, params.crop_width, params.crop_height
        if any([x > 0, y > 0, w > 0, h > 0]):
            try:
                w = w if w != -1 else image.size[0]
                h = h if h != -1 else image.size[1]
                image = image.crop((x, y, x + w, y + h))
            except Exception as e:
                logger.error(f"Image cropping failed: {e}")
                raise
        return image

    def make_ocr_tesseract(self, image: Image.Image):
        try:
            custom_oem_psm_config = r'--oem 3 --psm 6'
            return pytesseract.image_to_string(image, config=custom_oem_psm_config)
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""

    def _get_ocr_function(self, model):
        match model:
            case 'tesseract':
                return self.make_ocr_tesseract
            case 'doctr':
                return self.make_ocr_doctr
            case _:
                logger.error(f"Unsupported OCR model: {model}")
                raise ValueError(f"Unsupported OCR model: {model}")
