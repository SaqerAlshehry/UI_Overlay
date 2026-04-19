

# Model 
MODEL_PATH     = "vein_model.pt"   # change this when you get a new model
MODEL_CHANNELS = 1                 # 1 = segmentation only, 2 = segmentation + SDF

# Image dimensions (must match training dimensions) 
IMAGE_H = 704
IMAGE_W = 512

# Detection 
THRESHOLD = 0.65    # probability cutoff: above = vein, below = background

# Skeleton 
MIN_SEGMENT_LEN  = 12   # minimum segment length in pixels to keep
SPUR_PRUNE_ITERS = 8    # how many times to prune tiny branches

# --- IDSS Clinical Thresholds (DYNAMIC MILLIMETERS) ---

MIN_LENGTH_MM               = 16.0   # minimum vein length for IV insertion
MIN_DIAMETER_MM             = 2.44   # minimum vein diameter for catheter
MIN_BRANCH_DIST_MM          = 3.0    # minimum distance from branching point
MIN_USABLE_LENGTH_MM        = 16.0   # minimum usable segment length 
MIN_ENDPOINT_BRANCH_DIST_MM = 3.0    # min endpoint distance from bifurcation (~3mm)
MAX_TORTUOSITY              = 0.45   # maximum allowed curvature
MIN_EDGE_DIST_PX            = 20     # minimum distance from image border
MIN_CONFIDENCE              = 0.60   # minimum AI model confidence
MAX_CONF_VARIATION          = 0.15   # max confidence variation along segment (15%)

# Video optimization
REANALYZE_EVERY = 3   # run IDSS every 3 frames 