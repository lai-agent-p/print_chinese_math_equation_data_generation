from os.path import join
from os.path import dirname, abspath


class ProjectConfig:
    CONFIG_PATH = dirname(abspath(__file__))
    DATA_PATH = join(CONFIG_PATH, 'data')
    OUT_PATH = join(CONFIG_PATH, 'result')
    DELETE_OUT_PATH = join(CONFIG_PATH, 'deleted_images')
    OUT_SVG_IMAGE_PATH = join(CONFIG_PATH, 'img_out')
    SVG_PATH = join(CONFIG_PATH, 'svg_out')

    CHINESE_FONT_PATH = join(CONFIG_PATH, 'src/util/simsun.ttc')
    MATH_FONT_PATH = join(CONFIG_PATH, 'src/util/cambria.ttc')
    # MATH_FONT_PATH = join(CONFIG_PATH, 'src/util/ASANA.TTC')

    IM_FOLDERS_PATH = join(DATA_PATH, 'characters')
    CROHME_IM_FOLDERS_PATH = join(DATA_PATH, 'crohme_data')
    CASIA_CHARACTER_DATA = join(DATA_PATH, 'casia_character_dict.json')
    DELETION_SYMBOL_PATH = join(DATA_PATH, 'delete_ims')
    NEW_DELETION_SYMBOL_PATH = join(DATA_PATH, 'out_img')
    DELETION_OFFSET_PATH = join(DATA_PATH, 'offset_dict.json')

    # IM_FOLDERS_PATH = '/home/menglin/prepare_data/create_hand_im/data/characters'
    # CROHME_IM_FOLDERS_PATH = '/home/menglin/prepare_data/create_hand_im/data/crohme_data'
    # DELETION_SYMBOL_PATH = '/home/menglin/prepare_data/create_hand_im/data/delete_ims'
    # CASIA_CHARACTER_DATA = join('/home/menglin/prepare_data/create_hand_im/data/casia_character_dict.json')

    # LABELS_PATH = join(DATA_PATH, 'one_arg_labels.json')
    LABELS_PATH = join(DATA_PATH, 'full_label_new.json')
    # LABELS_PATH = join(DATA_PATH, 'root_labels.json')

    HAND_LEXICON_PATH = join(DATA_PATH, 'hand_lexicon.json')
    TYPE_LEXICON_PATH = join(DATA_PATH, 'type_lexicon.json')

    # DELETION_SYMBOL_PATH = join(DATA_PATH, 'delete_ims')

    TRAIN_LABELS_PATH = join(OUT_PATH, 'pretrain_train_labels.json')
    VALID_LABELS_PATH = join(OUT_PATH, 'pretrain_valid_labels.json')
    OUT_TRAIN_FOLDER_PATH = join(OUT_PATH, 'pretrain_train_ims')
    OUT_VALID_FOLDER_PATH = join(OUT_PATH, 'pretrain_valid_ims')
    
    INCOMPLETE_DELETION = False
    USE_LAYOUT_MODEL = False
    ONE_CHARACTER_SIZE_MODE = 'svg' # 'original' #'svg'
    IGNORE_SAMPLES_LIST = ['\\left', '\\right']
    IGNORE_CHAR_DELETION = ['2212', '2E', '22C5', '2C']
    OFFSET_POSITION = 'middle' #'up' #'middle'
    
    OUT_JSON_PATH = 'result/json'
    im_side = 64
    # SCALING_FACTOR = 0.1335
    SCALING_FACTOR = 0.05

    ascill_latex_pair = [('~', '\\sim'),
                        ('÷', '\\div'),
                        ('≠', '\\neq'),
                        ('<', '<'),
                        ('>', '>'),
                        ('≤', '\\leq'),
                        ('≥', '\\geq'),
                        ('=', '='),
                        ('|', '\\mid'),
                        ('-', '-'),
                        (',', ','),
                        (';', ';'),
                        (':', ':'),
                        ('!', '!'),
                        ('?', '?'),
                        ('…', '\\cdots'),
                        ('’', '\\prime'),
                        ('"', '"'),
                        ('“', '"'),
                        ('”', '"'),
                        ('(', '\\('),
                        (')', '\\)'),
                        ('[', '\\['),
                        (']', '\\]'),
                        ('{', '\\{'),
                        ('}', '\\}'),
                        ('$', '\\$'),
                        ('%', '\\%'),
                        ('+', '+'),
                        ('±', '\\pm'),
                        ('←', '\\leftarrow'),
                        ('→', '\\rightarrow'),
                        ('∴', '\\therefore'),
                        ('∵', '\\because'),
                        ('∈', '\\in'),
                        ('≌', '\\cong'),
                        ('∩', '\\cap'),
                        ('∏', '\\pi'),
                        ('⊥', '\\perp'),
                        ('∪', '\\cup'),
                        ('《', '《'),
                        ('》', '》'),
                        ('、', '、'),
                        ('△', '\\triangle'),
                        ('Δ', '\\Delta'),
                        ('○', '\\circ'),
                        ('⊙', '\\odot'),
                        ('∞', '\\infty'),
                        ('×', '\\times'),
                        ('≈', '\\approx'),
                        ('①', '\\textcircled { 1 }'),
                        ('②', '\\textcircled { 2 }'),
                        ('③', '\\textcircled { 3 }'),
                        ('④', '\\textcircled { 4 }'),
                        ('⑤', '\\textcircled { 5 }'),
                        ('⑥', '\\textcircled { 6 }'),
                        ('⑦', '\\textcircled { 7 }'),
                        ('⑧', '\\textcircled { 8 }'),
                        ('⑨', '\\textcircled { 9 }'),
                        ('℃', '^ { \\circ } C'),
                        ('∥', '\\parallel'),
                        ('。', '.'),
                        ('.', '.'),
                        ('□', '\\square'),
                        ('∠', '\\angle'),

                        ('∃', '\\exists'),
                        ('∀', '\\forall'),
                        ('σ', '\\sigma'),
                        ('γ', '\\gamma'),
                        ('λ', '\\lambda'),
                        ('μ', '\\mu'),
                        ('φ', '\\phi'),
                        ('θ', '\\theta'),
                        ('β', '\\beta'),
                        ('α', '\\alpha'),
                        ('/', '/')]

    crohme_keys = ['∃', '∀', 'σ', 'γ', 'λ', 'μ', 'φ', 'θ', 'β', 'α', '/', 'π']

    unicodes_convert = {}
    for pair in ascill_latex_pair:
        unicodes_convert[pair[0]] = pair[1]

    label2unicode = {}
    for key, val in unicodes_convert.items():
        if val not in label2unicode.keys():
            label2unicode[val] = [key]
        else:
            label2unicode[val].append(key)
    label2unicode['.'].append('dot')

    ONE_ARG_SYMBOLS = ['\\acute', '\\bar', '\\grave', '\\tilde', '\\wideparen', '\\vec', '\\dot', '\\ddot', '^', '_',
                       '\\hat', '\\textcircled', '\\RomanNumeralCaps', '\\romannumeral', '\\sqrt']
    TWO_ARG_SYMBOLS = ['\\frac']