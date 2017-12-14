BASE = "/home/bduser/unet/data/"
DATA_PATH = BASE+"/slices"
OUT_PATH  = BASE+"slices/Results/"
IMG_ROWS = 128
IMG_COLS = 128  
RESCALE_FACTOR = 1
SLICE_BY = 5 

BLOCKTIME = 0
NUM_INTER_THREADS = 2
NUM_INTRA_THREADS = 50

IN_CHANNEL_NO = 1
OUT_CHANNEL_NO = 1


EPOCHS = 10

MODEL_FN = "brainWholeTumor" #Name for Mode=1
#MODEL_FN = "brainActiveTumor" #Name for Mode=2
#MODEL_FN = "brainCoreTumor" #Name for Mode=3

#Use flair to identify the entire tumor: test reaches 0.78-0.80: MODE=1
#Use T1 Post to identify the active tumor: test reaches 0.65-0.75: MODE=2
#Use T2 to identify the active core (necrosis, enhancing, non-enh): test reaches 0.5-0.55: MODE=3
MODE=1

# Important that these are ordered correctly: [0] = master, [1] = worker, etc.
PS_HOSTS = ["10.100.68.245:2222"]
WORKER_HOSTS = ["10.100.68.193:2222"] #,"10.100.68.183:2222","10.100.68.185:2222","10.100.68.187:2222"]
