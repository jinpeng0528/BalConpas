from detectron2.config import CfgNode as CN


def add_continual_config(cfg):
    cfg.CONT = CN()
    cfg.CONT.TOT_CLS = 150
    cfg.CONT.BASE_CLS = 100
    cfg.CONT.INC_CLS = 10
    cfg.CONT.SETTING = 'overlapped'
    cfg.CONT.TASK = 1
    cfg.CONT.WEIGHTS = None
    cfg.CONT.OLD_WEIGHTS = None
    cfg.CONT.MED_TOKENS_WEIGHT = 1.0
    cfg.CONT.MEMORY = False