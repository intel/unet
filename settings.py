BASE = "/home/bduser/ge_tensorflow/data/"
DATA_PATH = BASE+"/slices"
OUT_PATH  = BASE+"slices/Results/"
IMG_ROWS = 128
IMG_COLS = 128  
RESCALE_FACTOR = 1
SLICE_BY = 5 


IN_CHANNEL_NO = 1
OUT_CHANNEL_NO = 1

EPOCHS = 6 # 10

MODEL_FN = "brainWholeTumor" #Name for Mode=1
#MODEL_FN = "brainActiveTumor" #Name for Mode=2
#MODEL_FN = "brainCoreTumor" #Name for Mode=3

#Use flair to identify the entire tumor: test reaches 0.78-0.80: MODE=1
#Use T1 Post to identify the active tumor: test reaches 0.65-0.75: MODE=2
#Use T2 to identify the active core (necrosis, enhancing, non-enh): test reaches 0.5-0.55: MODE=3
MODE=1
