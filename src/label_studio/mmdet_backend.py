# Copyright (c) OpenMMLab. All rights reserved.
# Modified from: https://github.com/open-mmlab/mmdetection/tree/main/projects/LabelStudio

import io
import json
import logging
import os
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import (DATA_UNDEFINED_NAME, get_image_size,
                                   get_single_tag_keys)
from label_studio_tools.core.utils.io import get_data_dir

from mmdet.apis import inference_detector, init_detector
from label_studio_converter import brush
from pycocotools import mask as maskUtils
import numpy as np
import pickle

logger = logging.getLogger(__name__)


class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection."""

    def __init__(self,
                 config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None,
                 score_threshold=0.5,
                 device='cpu',
                 **kwargs):

        super(MMDetection, self).__init__(**kwargs)
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file

        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(
            f'{self.__class__.__name__} reads images from {self.image_dir}')
        
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        print(json.dumps(self.parsed_label_config))
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(  # noqa E501
            self.parsed_label_config, 'BrushLabels', 'Image')
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)

        # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag # noqa E501
        self.labels_attrs = schema.get('labels_attrs')
        if self.labels_attrs:
            for i, (label, info) in enumerate(self.labels_attrs.items()):
                    self.label_map[i] = label

        print(f"label_map={self.label_map}")
        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold

    def _get_image_url(self, task):
        image_url = task['data'].get(
            self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3')
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={
                        'Bucket': bucket_name,
                        'Key': key
                    })
            except ClientError as exc:
                logger.warning(
                    f'Can\'t generate presigned URL for {image_url}. Reason: {exc}'  # noqa E501
                )
        return image_url

    def predict(self, tasks, **kwargs):
        return self._predict_segm(tasks, **kwargs)

    def _predict_segm(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)
        
        model_results = inference_detector(self.model,image_path).pred_instances
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)

        for item in model_results:
            masks, labels, scores = item["masks"], item['labels'], item['scores']
      
            output_labels = [self.label_map[int(cls)] for cls in labels]
            
            for i, mask in enumerate(masks):
                label = output_labels[i]
                if label not in self.labels_in_config:
                    print(label + ' label not found in project config.')
                    continue

                score = float(scores[i])
                if score < self.score_thresh:
                    continue

                if labels[i] == 1:
                    continue
                
                np_mask = mask.detach().cpu().numpy().astype(np.uint8)
                print(f'label={label}, output_label={label} mask={np_mask.shape}, mask_data={np.unique(np_mask)}, from_name={self.from_name}, to_name={self.to_name}')
                rle = brush.mask2rle(np_mask)

                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'brushlabels',
                    "original_width": img_width,
                    "original_height": img_height,
                    #"image_rotation": 0,
                    'value': {
                        'format': 'rle',
                        'rle': rle,
                        'brushlabels': [label],
                    },
                    'score': score
                })
                all_scores.append(score)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        return [{'result': results, 'score': avg_score}]

    def _predict_bbox(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)
        model_results = inference_detector(self.model,
                                           image_path).pred_instances
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        print(f'>>> model_results: {model_results}')
        print(f'>>> label_map {self.label_map}')
        print(f'>>> self.model.dataset_meta: {self.model.dataset_meta}')
        classes = self.model.dataset_meta.get('classes')
        print(f'Classes >>> {classes}')

        for item in model_results:
            print(f'item >>>>> {item}')
            bboxes, label, scores = item['bboxes'], item['labels'], item[
                'scores']
            score = float(scores[-1])
            if score < self.score_thresh:
                continue
            print(f'bboxes >>>>> {bboxes}')
            print(f'label >>>>> {label}')
            output_label = classes[list(self.label_map.get(label, label))[0]]
            print(f'>>> output_label: {output_label}')
            if output_label not in self.labels_in_config:
                print(output_label + ' label not found in project config.')
                continue

            for bbox in bboxes:
                bbox = list(bbox)
                if not bbox:
                    continue

                x, y, xmax, ymax = bbox[:4]
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': float(x) / img_width * 100,
                        'y': float(y) / img_height * 100,
                        'width': (float(xmax) - float(x)) / img_width * 100,
                        'height': (float(ymax) - float(y)) / img_height * 100
                    },
                    'score': score
                })
                all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        print(f'>>> RESULTS: {results}')
        return [{'result': results, 'score': avg_score}]


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data
