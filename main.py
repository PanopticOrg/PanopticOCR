from pydantic import BaseModel

from compute_ocr_task import ComputeOCRTask
from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface
from doctr.models import ocr_predictor

class PluginParams(BaseModel):
    """
    @ocr_prop_name: the name of the prop that will be created after an ocr
    """
    ocr_prop_name: str = "ocr"
  
  
class OCRPlugin(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):  
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params = PluginParams()  
        self.add_action_easy(self.ocr, ['execute'])
        self._model = ocr_predictor(pretrained=True, assume_straight_pages=True)

    async def ocr(self, context: ActionContext):
        commit = DbCommit()

        prop = await self.get_or_create_property('OCRPlugin', PropertyType.string, PropertyMode.sha1)
        commit.properties.append(prop)
        res = await self.project.do(commit)
        real_prop = res.properties[0]
        instances = await self.project.get_instances(ids=context.instance_ids)
        unique_sha1 = list({i.sha1: i for i in instances}.values())
        [await self.ocr_task(i, real_prop) for i in unique_sha1]

        return ActionResult(commit=res)

    async def ocr_task(self, instance, prop):
        task = ComputeOCRTask(self, instance, prop)
        self.project.add_task(task)