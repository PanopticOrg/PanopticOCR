from pydantic import BaseModel
from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image


class PluginParams(BaseModel):
    """
    @ocr_prop_name: the name of the prop that will be created after an ocr
    @crop_x, crop_y, crop_width, crop_height: Optional cropping parameters
    """
    ocr_prop_name: str = "ocr"
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0

  
class OCRPlugin(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):  
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params = PluginParams()  
        self.add_action_easy(self.ocr, ['execute'])
        self._model = ocr_predictor(pretrained=True, assume_straight_pages=True)

    async def ocr(self, context: ActionContext):
        commit = DbCommit()

        prop = await self.get_or_create_property('OCRPlugin', PropertyType.string, PropertyMode.sha1)

        instances = await self.project.get_instances(ids=context.instance_ids)
        unique_sha1 = list({i.sha1: i for i in instances}.values())
        tasks = [await self.single_ocr(i, prop) for i in unique_sha1]
        values = tasks

        commit.properties.append(prop)
        commit.image_values.extend(values)

        res = await self.project.do(commit)
        return ActionResult(commit=res)

    async def single_ocr(self, instance: Instance, prop: Property):
        text = await self.make_ocr(instance.url)
        return ImageProperty(property_id=prop.id, sha1=instance.sha1, value=text)

    async def make_ocr(self, image_path):
        # Ouvrir l’image avec PIL pour appliquer le crop
        with Image.open(image_path) as img:
            if self.params.crop_width > 0 and self.params.crop_height > 0:
                x, y, w, h = self.params.crop_x, self.params.crop_y, self.params.crop_width, self.params.crop_height
                img = img.crop((x, y, x + w, y + h))

            # Convertir en format utilisable par Doctr
            single_img_doc = DocumentFile.from_images(img)

        # Effectuer l'OCR
        result = self._model(single_img_doc)

        # Extraire le texte reconnu avec un seuil de confiance de 0.50
        res = ""
        js = result.export()
        for block in js['pages'][0]['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    if word['confidence'] > 0.50:
                        res += " " + word['value']
        
        return res
