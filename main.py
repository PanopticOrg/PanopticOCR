from pydantic import BaseModel

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

from doctr.io import DocumentFile
from doctr.models import ocr_predictor


class PluginParams(BaseModel):
    """
    @ocr_prop_name: the name of the prop that will be created after an ocr
    """
    ocr_prop_name: str = "ocr"
  
  
class OCRPlugin(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):  
        super().__init__(name=name,project=project, plugin_path=plugin_path)
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
        single_img_doc = DocumentFile.from_images(image_path)
        result = self._model(single_img_doc)
        res = ""
        js = result.export()
        for block in js['pages'][0]['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    if word['confidence'] > 0.50:
                        res += " " + word['value']
        return res
