from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytesseract

if TYPE_CHECKING:
    from . import OCRPlugin

from PIL import Image

from panoptic.core.task.task import Task
from panoptic.models import Instance, Property, ImageProperty, DbCommit

logger = logging.getLogger('PanopticOCR:OCRTask')


class ComputeOCRTask(Task):
    def __init__(self, plugin: OCRPlugin, instance: Instance, prop:Property, model: str):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.instance = instance
        self.name = f'Compute OCR'
        self.prop = prop
        self.model = model

    async def run(self):
        instance = self.instance
        ocr_function = self._get_ocr_function(self.model)
        image = self.preprocess_image(instance.url)
        text = await self._async(ocr_function, image)
        commit = DbCommit()
        commit.image_values.extend([ImageProperty(property_id=self.prop.id, sha1=instance.sha1, value=text)])
        res = await self.project.do(commit)
        # use these two lines to update on front when the data is available
        self.project.ui.commits.append(commit)
        self.project._project.db.on_import_instance.emit(commit.image_values[0])
        return res.image_values[0]

    def make_ocr_doctr(self, image: Image):
        image_arr = np.asarray(image)
        result = self.plugin._model([image_arr])
        res = ""
        js = result.export()
        for block in js['pages'][0]['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    if word['confidence'] > 0.30:
                        res += " " + word['value']
        return res

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        params = self.plugin.params
        x, y, w, h = params.crop_x, params.crop_y, params.crop_width, params.crop_height
        if any([x > 0, y > 0, w > 0, h > 0]):
            w = w if w != -1 else image.size[0]
            h = h if h != -1 else image.size[1]
            image = image.crop((x, y, x + w, y + h))
        return image

    def make_ocr_tesseract(self, image: Image):
        custom_oem_psm_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(image, config=custom_oem_psm_config)

    def _get_ocr_function(self, model):
        match model:
            case 'tesseract':
                return self.make_ocr_tesseract
            case 'doctr':
                return self.make_ocr_doctr
            case _:
                return self.make_ocr_doctr
