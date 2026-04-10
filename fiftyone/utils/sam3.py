"""
`Segment Anything 3 <https://github.com/facebookresearch/sam3>`_
wrapper for the FiftyOne Model Zoo.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging
import os
from enum import Enum

import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
import fiftyone.utils.torch as fout
import fiftyone.utils.sam as fosam
import fiftyone.utils.sam2 as fosam2

fou.ensure_torch()
import torch

sam3 = fou.lazy_import("sam3")
sam3tr = fou.lazy_import("sam3.train.transforms.basic_for_api")
sam3ds = fou.lazy_import("sam3.train.data.sam3_image_dataset")

logger = logging.getLogger(__name__)


class SegmentAnything3ImageModelConfig(fosam.SegmentAnythingModelConfig):
    """Configuration for running a :class:`SegmentAnything3ImageModel`.

    See :class:`fiftyone.utils.torch.TorchImageModelConfig` for additional
    arguments.

    Args:
        classes (None): a list of custom classes for use as SAM3 text prompts
    """

    def __init__(self, cfg_dict):
        """Initializes :class:`SegmentAnythingModelConfig`

        Args:
            cfg_dict: a dictionary with config parameters
        """
        d = self.init(cfg_dict)
        super().__init__(d)
        self.get_item_cls = self.parse_string(
            d,
            "get_item_cls",
            default="fiftyone.utils.sam3.SegmentAnything3ImageGetItem",
        )
        self.classes = self.parse_array(d, "classes", default=None)
        self.operation_mode = self.parse_string(
            d, "operation_mode", default="concept"
        )


class _SAM3Predictor(fosam2._SAM2Predictor):
    def __init__(self, model):
        if model.inst_interactive_predictor is None:
            raise AttributeError(
                "Sam3Image.inst_interactive_predictor must be initialized."
            )

        self.processor = model.inst_interactive_predictor
        self.image_id = None

    def image_transform(self, img):
        """Transforms image for SAM3 model input.

        Args:
            img: a PIL image

        Returns:
            a PIL image
            a tuple containing original image dimensions
        """
        # SAM2 does image pre-processing when it calls SAM2ImagePredictor.set_image.
        # No straight-forward way to decouple them other than extracting the functionality.
        return img, img.size[::-1]


class SegmentAnything3ImageGetItem(fosam.SegmentAnythingImageGetItem):
    """A :class:`GetItem` that loads images, bounding boxes and/or keypoints to feed to
    :class:`SegmentAnythingModel` instances.

    Args:
        field_mapping (None): the user-supplied dict mapping keys in
            :attr:`required_keys` to field names of their dataset that contain
            the required values
        transform (None): SAM specific image transform function to apply
        use_numpy (False): whether to use numpy arrays rather than PIL images
            and Torch tensors when loading data
        box_transform (None): SAM specific box transform function to apply
        point_transform (None): SAM specific point transform function to apply
        text_prompts (None): Text prompts for concept prompting the model
        operation_mode ("concept"): Operation mode of the model (required for collate_fn)
    """

    def __init__(
        self,
        field_mapping=None,
        transform=None,
        use_numpy=False,
        box_transform=None,
        point_transform=None,
        text_prompts=None,
        operation_mode="concept",
        **kwargs,
    ):
        super().__init__(
            field_mapping=field_mapping,
            transform=transform,
            use_numpy=use_numpy,
            box_transform=box_transform,
            point_transform=point_transform,
            **kwargs,
        )
        self.text_prompts = text_prompts
        self.operation_mode = operation_mode

    def __call__(self, d):
        """Prepares the model input for a given sample's data.

        Args:
            d: a dict mapping the :meth:`required_keys` to values from the
                sample being processed

        Returns:
            the model input
        """
        item_dict = super().__call__(d=d)
        item_dict["operation_mode"] = self.operation_mode
        if self.operation_mode == "concept":
            item_dict["text_prompts"] = self.text_prompts
        return item_dict


def build_sam_datapoint_transform():
    transform = sam3tr.ComposeAPI(
        transforms=[
            sam3tr.RandomResizeAPI(
                sizes=1008,
                max_size=1008,
                square=True,
                consistent_transform=False,
            ),
            sam3tr.ToTensorAPI(),
            sam3tr.NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform


# In SAM3, when operating in "concept" mode, (text, box and point prompts) are used via datapoints and get collated as datapoints.
# When operating in "visual" mode, (box, point prompts) are used via SAM2 forward_pass
class SegmentAnything3ImageModel(fosam.SegmentAnythingModel):
    """Wrapper for running `Segment Anything 3 <https://ai.meta.com/research/sam3>`_
    inference.

    Args:
        config: a :class:`SegmentAnything3ModelConfig`
    """

    def __init__(self, config):
        if config.output_processor_cls is None:
            config.output_processor_cls = (
                "fiftyone.utils.sam3.SAM3SegmenterOutputProcessor"
            )

        if config.entrypoint_args is None:
            config.entrypoint_args = {}
        if "enable_inst_interactivity" not in config.entrypoint_args:
            # Sam3Image.inst_interactive_predictor is only needed for "visual" operation mode.
            # Always set to True for easy switching of operation modes in a loaded zoo model.
            config.entrypoint_args["enable_inst_interactivity"] = True

        fout.TorchImageModel.__init__(self, config)
        self._sam_auto_generator = None
        self._sam_predictor = self._load_predictor()
        self._operation_mode = self.config.operation_mode

    def _load_predictor(self):
        return _SAM3Predictor(model=self._model)

    def _load_model(self, config):
        if "device" not in config.entrypoint_args:
            config.entrypoint_args["device"] = self._device
        return super()._load_model(config)

    def _download_model(self, config):
        # Download sam3 to fo.config.model_zoo_dir from HF hub.
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="facebook/sam3",
            filename=os.path.basename(config.model_path),
            local_dir=os.path.dirname(config.model_path),
            local_dir_use_symlinks=False,
        )

    @property
    def operation_mode(self):
        """Whether to use the model in visual or concept segmentation mode"""
        return self._operation_mode

    @operation_mode.setter
    def operation_mode(self, value):
        self._operation_mode = value

    @staticmethod
    def collate_fn(batch):
        """Collates a batch of inputs where each input is generated from :class:`SegmentAnything3ImageGetItem`.

        Args:
            batch: a list of dict containing model input from :class:`SegmentAnything3ImageGetItem`

        Returns:
            a collated dictionary of model input for the batch.
        """
        results = fosam.SegmentAnythingModel.collate_fn(batch)

        operation_modes = results["operation_mode"]
        if not all(op == operation_modes[0] for op in operation_modes):
            raise ValueError(
                "All samples in a batch must have the same operation_mode"
            )
        results["operation_mode"] = results["operation_mode"][0]

        if results["operation_mode"] == "visual":
            return results

        # For "concept" mode, the collate output needs to be in Datapoint.
        transform = build_sam_datapoint_transform()
        datapoints = []
        for img_idx in range(len(results["image"])):
            datapoint = sam3ds.Datapoint(find_queries=[], images=[])
            datapoint.images = [
                sam3ds.Image(
                    data=results["image"][img_idx],
                    objects=[],
                    size=results["original_size"][img_idx],
                )
            ]
            text_prompts = (
                results["text_prompts"][img_idx]
                if "text_prompts" in results
                else []
            )
            for tx in text_prompts:
                datapoint.find_queries.append(
                    sam3ds.FindQueryLoaded(
                        query_text=tx,
                        image_id=0,
                        object_ids_output=[],  # unused for inference
                        is_exhaustive=True,  # unused for inference
                        query_processing_order=0,
                        inference_metadata=sam3ds.InferenceMetadata(
                            original_image_id=img_idx,
                            original_size=results["original_size"][::-1],
                            # dummy values
                            coco_image_id=img_idx,
                            original_category_id=1,
                            object_id=0,
                            frame_index=0,
                        ),
                    )
                )
            box_prompts = (
                results["boxes_xyxy"][img_idx]
                if "boxes_xyxy" in results
                else None
            )
            if box_prompts is not None:
                datapoint.find_queries.append(
                    sam3ds.FindQueryLoaded(
                        query_text="visual",
                        image_id=0,
                        object_ids_output=[],  # unused for inference
                        is_exhaustive=True,  # unused for inference
                        query_processing_order=0,
                        input_bbox=torch.tensor(
                            box_prompts, dtype=torch.float
                        ),
                        input_bbox_label=torch.tensor(
                            results["boxes_labels"][img_idx], dtype=torch.bool
                        ),
                        inference_metadata=sam3ds.InferenceMetadata(
                            original_image_id=img_idx,
                            original_size=results["original_size"][::-1],
                            # dummy values
                            coco_image_id=img_idx,
                            original_category_id=1,
                            object_id=0,
                            frame_index=0,
                        ),
                    )
                )
            datapoints.append(transform(datapoint))
        return sam3.train.data.collator.collate_fn_api(
            datapoints, dict_key="datapoints"
        )

    def build_get_item(self, field_mapping=None):
        """Builds a :class:`SegmentAnything3ImageGetItem` for loading model input from samples.

        Args:
            field_mapping (None): a dict mapping required keys to sample fields

        Returns:
            a :class:`SegmentAnything3ImageGetItem` instance
        """
        rm_text_prompts, rm_op_mode = False, False
        if self.config.get_item_args is None:
            self.config.get_item_args = {}
        if (
            "text_prompts" not in self.config.get_item_args
            and self.config.classes is not None
        ):
            self.config.get_item_args["text_prompts"] = self.config.classes
            rm_text_prompts = True
        if "operation_mode" not in self.config.get_item_args:
            self.config.get_item_args[
                "operation_mode"
            ] = self.config.operation_mode
            rm_op_mode = True
        self.config.get_item_args["use_numpy"] = False
        get_item = super().build_get_item(field_mapping=field_mapping)
        if rm_text_prompts:
            _ = self.config.get_item_args.pop("text_prompts")
        if rm_op_mode:
            _ = self.config.get_item_args.pop("operation_mode")
        return get_item

    def _forward_pass(self, imgs):
        raise NotImplementedError("TODO")

    def _forward_pass_auto(self, imgs):
        raise RuntimeError(
            "sam3.model.sam3_image.Sam3Image doesn't support auto segmentation. You may use one of the SAM/SAM2 zoo models for auto segmentation."
        )


class SAM3SegmenterOutputProcessor(fosam.SAMSegmenterOutputProcessor):
    pass
