from os.path import join, dirname, abspath


class ProjectConfig():
    def __init__(self):
        self.root_path = dirname(abspath(__file__))
        self.MODEL_NAME = 'LayoutLearning'
        self.PRETRAIN_MODEL_NAME = 'DenseNetPretrain'
        self.DICT_SIZE = 3894
        self.PRETRAIN_IMAGE_FOLDER = join(self.root_path, 'data/characters')
        self.TRAIN_IMAGE_FOLDER = join(self.root_path, 'data/train_ims')
        self.TRAIN_META_PATH = join(self.root_path, 'data/labels.json')
        self.CONTENT_LABEL_PATH = join(self.root_path, 'data/contents.json')
        self.TEST_FOLDER_PATH = 'data/test_ims'
        self.TEST_OUT_PATH = 'eval_out/'

        self.DOWN_SMALL_SYMBOLS = {',', '.', '、', '。'}
        self.UP_SMALL_SYMBOLS = {'"', '\\prime', '`', '‘', '’', '"', '“', '”'}

        self.h_mean, self.h_std = 81.5176804299062, 17.84207107231951
        self.w_mean, self.w_std = 70.00431321494372, 17.279074558835312

        self.gap_shift_mean, self.gap_shift_std = 0, 10.881739353437661
        self.center_shift_mean, self.center_shift_std = 0.9783846733225106, 10.881739353437661

        self.slope_mean, self.slope_std = -0.004690826736476126, 0.04094917547579227
        self.base_gap_mean, self.base_gap_std = 8.617104586099268, 9.300101013712046
        self.DO_TEACHER_FORCE = False
        self.USE_SAMPLE_LABELS = True
        self.LEARN_GAUSSIAN_DIST = True
        self.ENCODER_OUT_DIM = 128
        self.N_PI = 3

        self.BATCH_SIZE = 10
        self.PRETRAIN_BATCH_SIZE = 128
        self.LEXICON_PATH = join(self.root_path, 'data/lexicon.json')
        self.OPTIMIZER = 'Adam'
        self.USE_GPU = True
        self.N_VOCAB = 416
        self.N_EPOCHES = 10000
        self.in_im_height = 32
        self.in_im_width = 32
        self.RECORD_ITERATIONS = 100
        self.WEIGHT_DECAY = 1e-5
        self.INIT_LEARNING_RATE = 1e-3
        self.LR_DECAY_RATE = 0.96
        self.CLIP_NORM = 100.
        self.MAX_BATCH_SIZE = 64
        self.CHUNK_SIZE = 50
        self.STEP_SIZE = 3000
        self.CLIP_NORM = 100.

        self.CHECK_POINT_FOLDER = join(self.root_path, 'checkpoints')
        self.PRETRAIN_CHECK_POINT_FOLDER = join(self.root_path, 'pretrain_checkpoints')
        self.LOG_DIR = join(self.root_path, 'logs')
        self.PRETRAIN_LOG_DIR = join(self.root_path, 'pretrain_logs')
        self.TRAIN_VIS_PATH = join(self.root_path, 'train_vis')