import math
import os.path as osp
from typing import Optional, Sequence

from mmengine import FileClient
from mmengine.hooks import Hook
from mmengine.runner import EpochBasedTrainLoop, Runner
from mmengine.visualization import Visualizer

from mmaction.registry import HOOKS
from mmaction.structures import ActionDataSample


@HOOKS.register_module()
class NceWeightModifyHook(Hook):

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch,
                         outputs) -> None:
        if runner.iter==1200 and runner.epoch==0:
            runner.model.module.pretrain['stan_text_nce'] = 1
            
@HOOKS.register_module()
class MaskModifyHook(Hook):

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch,
                         outputs) -> None:
        iter = runner.iter % 3000
        if iter < 2000:
            runner.model.module.backbone.mask = 'tube'
        else:
            runner.model.module.backbone.mask = None