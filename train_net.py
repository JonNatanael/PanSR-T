# ------------------------------------------------------------------------
# PanSR training script (LaRS panoptic segmentation).
# Built on MaskDINO / Mask2Former training scripts (Apache-2.0).
# ------------------------------------------------------------------------
import warnings

try:
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except Exception:
    pass

# PyTorch 2.4's checkpoint implementation still calls the deprecated CPU AMP
# spelling internally.  Keep this filter deliberately narrow so other future
# compatibility warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cpu\.amp\.autocast.*is deprecated.*",
    category=FutureWarning,
    module=r"torch\.utils\.checkpoint",
)

import logging
import os
import random
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from pansr import add_pansr_config
from pansr.temporal.config import add_temporal_config


from pansr.trainer import Trainer  # noqa: E402  (extracted for reuse by temporal trainer)


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)  # for the poly LR schedule
    add_pansr_config(cfg)
    add_temporal_config(cfg)  # idempotent; TEMPORAL node for TG-PanSR configs
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="pansr")
    return cfg


def _select_trainer(cfg):
    """Vanilla PanSR configs use the original Trainer; temporal (TG-PanSR)
    configs use TemporalTrainer (temporal loader + parameter groups +
    gradient accumulation)."""
    if cfg.MODEL.META_ARCHITECTURE == "TemporalPanSR":
        from pansr.temporal.trainer import TemporalTrainer

        return TemporalTrainer
    return Trainer


def main(args):
    cfg = setup(args)
    trainer_cls = _select_trainer(cfg)
    if args.eval_only:
        model = trainer_cls.build_model(cfg)
        from pansr.checkpoint_utils import load_model_weights

        load_model_weights(model, cfg, resume=args.resume)
        return trainer_cls.test(cfg, model)

    trainer = trainer_cls(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()
    port = random.randint(1000, 20000)
    args.dist_url = 'tcp://127.0.0.1:' + str(port)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
