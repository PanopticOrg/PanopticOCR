from pydantic import BaseModel
import logging

from compute_ocr_task import ComputeOCRTask
from panoptic.core.plugin.plugin import APlugin
from panoptic.models import ActionContext, PropertyType, PropertyMode, DbCommit, Instance, ImageProperty, Property
from panoptic.models.results import ActionResult
from panoptic.core.plugin.plugin_project_interface import PluginProjectInterface

logger = logging.getLogger(__name__)

class PluginParams(BaseModel):
    """
    @ocr_prop_name: the name of the prop that will be created after an ocr
    @model: the ocr model to use, doctr works on more complex cases but is slower,
            for tesseract you need to install tesseract manually but it's faster
    @crop_x: x to start cropping the image, 0 is left
    @crop_y: y to start cropping the image, 0 is top
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
        try:
            commit = DbCommit()

            try:
                prop = await self.get_or_create_property(
                    self.params.ocr_prop_name,
                    PropertyType.string,
                    PropertyMode.sha1
                )
                commit.properties.append(prop)
                res = await self.project.do(commit)
                real_prop = res.properties[0]
            except Exception as e:
                logger.error(f"Failed to create or fetch property: {e}")
                raise RuntimeError("OCR property creation failed.")

            try:
                instances = await self.project.get_instances(ids=context.instance_ids)
                unique_sha1 = list({i.sha1: i for i in instances}.values())
            except Exception as e:
                logger.error(f"Failed to fetch instances: {e}")
                raise RuntimeError("Failed to fetch instance data.")

            if self.params.model == "doctr":
                try:
                    self._model = await self._init_doctr_model()
                except Exception as e:
                    logger.error(f"Failed to initialize doctr model: {e}")
                    raise RuntimeError("OCR model initialization failed.")

            try:
                for instance in unique_sha1:
                    await self.ocr_task(instance, real_prop)
            except Exception as e:
                logger.error(f"Error while scheduling OCR tasks: {e}")
                raise RuntimeError("Failed to schedule OCR tasks.")

            return ActionResult(commit=res)

        except Exception as e:
            logger.exception(f"OCR action failed: {e}")
            raise e  # rethrow for Panoptic to handle/report as needed

    async def ocr_task(self, instance, prop):
        try:
            task = ComputeOCRTask(self, instance, prop, self.params.model)
            self.project.add_task(task)
        except Exception as e:
            logger.error(f"Failed to add OCR task for instance {instance.id}: {e}")
            raise

    async def _init_doctr_model(self):
        from doctr.models import ocr_predictor
        return ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_mobilenet_v3_small",
            pretrained=True,
            assume_straight_pages=True,
            disable_page_orientation=True,
            disable_crop_orientation=True
        )
