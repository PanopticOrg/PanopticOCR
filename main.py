from pydantic import BaseModel

from .compute_ocr_task import ComputeOCRTask

from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult, Notif, NotifType
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

class PluginParams(BaseModel):
    """
    @ocr_prop_name: the name of the prop that will be created after an ocr
    @model: the ocr model to use, doctr works on more complex cases but is slower, for tesseract you need to install tesseract manually but it's faster
    @crop_x: x to start croping the image, 0 is left
    @crop_y: y to start croping the image 0 is top
    @crop_width: crop width, -1 to use full width
    @crop_height: crop height, -1 to use full height
    """
    ocr_prop_name: str = "ocr"
    model: str = "doctr"
    crop_x: int = 0
    crop_y: int = 0
    crop_width: int = 0
    crop_height: int = 0

  
class OCRPlugin(APlugin):  
    def __init__(self, project: PluginProjectInterface, plugin_path: str, name: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params = PluginParams()  
        self.add_action_easy(self.ocr, ['execute'])
        self._model = None

    async def ocr(self, context: ActionContext):
        commit = DbCommit()

        prop = await self.project.get_or_create_property(self.params.ocr_prop_name, PropertyType.string, PropertyMode.sha1)
        commit.properties.append(prop)
        res = await self.project.do(commit)
        real_prop = res.properties[0]
        instances = await self.project.get_instances(ids=context.instance_ids)
        unique_sha1 = list({i.sha1: i for i in instances}.values())
        if self.params.model == "doctr":
            self._model = await self._init_doctr_model()
        [await self.ocr_task(i, real_prop) for i in unique_sha1]

        notif = Notif(
            type=NotifType.INFO,
            name='compute_ocr',
            message=f'OCR started on {len(unique_sha1)} images'
        )
        return ActionResult(notifs=[notif])

    async def ocr_task(self, instance, prop):
        task = ComputeOCRTask(self, instance, prop, self.params.model)
        self.project.add_task(task)

    async def _init_doctr_model(self):
        from doctr.models import ocr_predictor
        return  ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_mobilenet_v3_small", pretrained=True, assume_straight_pages=True,
                      disable_page_orientation=True, disable_crop_orientation=True)