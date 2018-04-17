import os

BASE = "../../"
OUT_PATH  = os.path.join(BASE, "data/")

IMG_ROWS = 128
IMG_COLS = 128

IN_CHANNEL_NO = 1
OUT_CHANNEL_NO = 1

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
PRINT_MODEL = False

# Mode 1: Use flair to identify the entire tumor
# Mode 2: Use T1 Gd to identify the active tumor
# Mode 3: Use T2 to identify the active core (necrosis, enhancing, non-enh)
MODE=1  # 1, 2, or 3


import psutil
BLOCKTIME = 0
NUM_INTER_THREADS = 1
NUM_INTRA_THREADS = psutil.cpu_count(logical=False) - 2

CHANNELS_FIRST = False
USE_KERAS_API = False
USE_UPSAMPLING = False
CREATE_TRACE_TIMELINE=False
