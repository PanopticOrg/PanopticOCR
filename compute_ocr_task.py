from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from . import OCRPlugin

from PIL import Image

from panoptic.core.task.task import Task
from panoptic.models import Instance, Vector, Property, ImageProperty, DbCommit

logger = logging.getLogger('PanopticOCR:OCRTask')


class ComputeOCRTask(Task):
    def __init__(self, plugin: OCRPlugin, instance: Instance, prop:Property):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.instance = instance
        self.name = f'Compute OCR'
        self.prop = prop

    async def run(self):
        instance = self.instance
        text = await self._async(self.make_ocr, instance.url)
        commit = DbCommit()
        commit.image_values.extend([ImageProperty(property_id=self.prop.id, sha1=instance.sha1, value=text)])
        res = await self.project.do(commit)
        # use these two lines to update on front when the data is available
        self.project.ui.commits.append(commit)
        self.project._project.db.on_import_instance.emit(commit.image_values[0])
        return res.image_values[0]

    def make_ocr(self, image_path):
        # single_img_doc = DocumentFile.from_images(image_path)
        image = Image.open(image_path)
        params = self.plugin.params
        if params.crop_x > 0:
            x, y, w, h = params.crop_x, params.crop_y, params.crop_width, params.crop_height
            image = image.crop((x, y, x + w, y + h))
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

