"""
This is for helper constant declaration

MCCT (initial) / Minnie Cherry Chua Tan  8-Jul-21 added main processing constant
MCCT (initial) / Minnie Cherry Chua Tan 24-Jul-21 added HAS_Y
MCCT (initial) / Minnie Cherry Chua Tan 06-Aug-21 rename from SPLIT_TEST to TEST_SIZE for consistency and recallability,
    added MAX_CARDINALITY
MCCT (initial) / Minnie Cherry Chua Tan 21-Aug-21 Added RANDOM_STATE
MCCT (initial) / Minnie Cherry Chua Tan 21-Aug-21 Added INDEX, LAST_SCORE, SCORES
MCCT (initial) / Minnie Cherry Chua Tan 28-Aug-21 Added DEGREE, X
MCCT (initial) / Minnie Cherry Chua Tan 01-Sep-21 update SCORES to REGRESSION_SCORES, add CLASSIFICATION_SCORES
MCCT (initial) / Minnie Cherry Chua Tan 02-Sep-21 Added PREDICT_TEST to make test sets va input file, SMOOTH, SMOOTH_VALUE
MCCT (initial) / Minnie Cherry Chua Tan 04-Sep-21 Added STATS
MCCT (initial) / Minnie Cherry Chua Tan 07-Sep-21 Added CLASSIFICATION_REPORT, CONFUSION_MATRIX, PREDICTION_ERROR, ROC_AUC
MCCT (initial) / Minnie Cherry Chua Tan 15-Sep-21 Added CLUSTER
MCCT (initial) / Minnie Cherry Chua Tan 16-Sep-21 Added Y
MCCT (initial) / Minnie Cherry Chua Tan 16-Sep-21 Added CLUSTER_SCORES
MCCT (initial) / Minnie Cherry Chua Tan 22-Sep-21 Added INTERCLUSTER_DISTANCE, SILHOUETTE
MCCT (initial) / Minnie Cherry Chua Tan 22-Oct-21 Added RULES
MCCT (initial) / Minnie Cherry Chua Tan 29-Oct-21 Added UCB, THOMPSONSAMPLING, REINFORCEMENT_LEARNING, SAMPLING
MCCT (initial) / Minnie Cherry Chua Tan 05-Nov-21 Added STOP_WORDS, TSV, TEXT
MCCT (initial) / Minnie Cherry Chua Tan 11-Nov-21 Added ANN, NEURAL_NETWORK
MCCT (initial) / Minnie Cherry Chua Tan 14-Nov-21 Added NEURAL_SCORES
MCCT (initial) / Minnie Cherry Chua Tan 30-Jan-22 Added KERAS
MCCT (initial) / Minnie Cherry Chua Tan 03-Feb-22 Added GENERATED
MCCT (initial) / Minnie Cherry Chua Tan 08-Feb-22 Added CNN, Updated NEURAL
MCCT (initial) / Minnie Cherry Chua Tan 01-Mar-22 Added ENCODING, COLUMNS
MCCT (initial) / Minnie Cherry Chua Tan 02-Mar-22 Added MAX_VOCAB_SIZE
MCCT (initial) / Minnie Cherry Chua Tan 10-Mar-22 Added BASE, TRAIN_FILE, TEST_FILE
MCCT (initial) / Minnie Cherry Chua Tan 11-Mar-22 Added IMAGE, FILENAME, PREDICT_FILE
MCCT (initial) / Minnie Cherry Chua Tan 17-Mar-22 Updated TRAIN_FILE to TRAIN_DIR, TEST_FILE to TEST_DIR
MCCT (initial) / Minnie Cherry Chua Tan 18-Mar-22 Added RNN, LSTM, TRAIN_FILE, TEST_FILE, SPLIT_FILE
MCCT (initial) / Minnie Cherry Chua Tan 19-Mar-22 Added TIMESTEPS, INTERVAL
MCCT (initial) / Minnie Cherry Chua Tan 25-Mar-22 Added Y_PREDICT
MCCT (initial) / Minnie Cherry Chua Tan 02-Apr-22 Added SOM, UNSUPERVISED, NONSUPERVISED
MCCT (initial) / Minnie Cherry Chua Tan 15-Apr-22 Added RBM, DELIMITER
MCCT (initial) / Minnie Cherry Chua Tan 16-Apr-22 Added HEADER, HIDDEN, EPOCH, BATCH_SIZE
MCCT (initial) / Minnie Cherry Chua Tan 16-Apr-22 Added LOSS, ACCURACY
MCCT (initial) / Minnie Cherry Chua Tan 29-Apr-22 Added MODEL_ADD, X_SHAPE
"""


# MCCT/MCT/MT is the initial shortname name of Minnie Cherry Chua Tan - same person  as Minnie Tan,
# without second name (Cherry) and middle name (Chua) from my mother (Melody Chua)
# and my father's surname (Julio Tan with Chinese's name Lo Cho Hui), my aut networkID is xrx5385 when I was student
__author__ = 'Minnie Tan'

from typing import Union, List, TypeVar

class Constant:
    # type hinting / definition
    PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')
    NUM = Union[int, float]
    PRIMITIVE = Union[int, float, str, list]
    BINARY = (0, 1)
    T = TypeVar('T') # generic type like object

    # main processing
    CONFIG_FILE = "config.json"
    DETAIL = "detail"
    FOLDER = "folder"
    MODEL = "Model"
    FILE = "File"
    PREP_ARGS = "PreprocessorParams"
    MODEL_ARGS = "ModelParams"
    GRAPH_ARGS = "GraphParams"
    GROUP_BY = 5
    INDEX = "index"
    LAST = "last"
    IS_LAST = "is_last"
    STATS = "stats"

    # configuration constant values
    TEST_SIZE = "test_size"
    GRAPH = "graph"
    SAVE = "save"
    FILE_STATS = "file_stats"
    BASE = "base"
    INPUT = "input"
    OUTPUT = "output"
    LOG = "log"
    LOGGING = 'logging'
    MAX_CARDINALITY = "max_cardinality"
    REGRESSION_SCORES = 'regression_scores'
    CLASSIFICATION_SCORES = 'classification_scores'
    CLUSTER_SCORES = "cluster_scores"
    NEURAL_SCORES = "neural_scores"

    CLASSIFICATION_TYPE = "classification_type"
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CLUSTER = "cluster"
    RULES = "rules"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NEURAL_NETWORK = "neural_network"
    UNSUPERVISED = "unsupervised"

    # file constant
    FILENAME = "filename"
    TRAIN_DIR = "train_dir"
    TEST_DIR = "test_dir"
    TRAIN_FILE = "train_file"
    TEST_FILE = "test_file"
    SPLIT_FILE = "split_file"

    PREDICT_FILE = "predict_file"

    # preprocessor constant
    METHOD = "method"
    PARAMS = "params"
    HAS_Y = "has_y"
    STOP_WORDS = "stop_words"
    RANDOM_STATE = "random_state"
    KEY = "key"
    VALUE = "value"
    METHOD_CALL = "method_call"
    SCALER = "scaler"
    FEATURE = "feature"
    IMPUTER = "imputer"
    TEXT = "text"
    X = "x"
    y = "y"
    Y_PREDICT = "y-predicted"
    PREDICT_TEST = "predict_test"
    ENCODING = "encoding"
    DELIMITER = "delimiter"
    HEADER = "header"
    COLUMNS = "columns"
    IMAGE = "image"
    TIMESTEPS = "timesteps"
    INTERVAL = "interval"
    MODEL_ADD = "model_add"
    X_SHAPE = "x_shape"

    # model constant
    DEGREE = "degree"
    HIDDEN = "hidden"
    EPOCH = "epoch"
    LOSS = "loss"
    ACCURACY = "accuracy"
    BATCH_SIZE = "batch_size"

    MIN_SUPPORT = "min_support"
    MIN_CONFIDENCE = "min_confidence"
    MIN_LIFT = "min_lift"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    ITEM = "item"
    METRICS = "metrics"
    SUPPORT = "support"
    CONFIDENCE = "confidence"
    LIFT = "lift"
    APRIORI = "apriori"
    ECLAT = "eclat"

    # modifiers names
    UCB = "UCB" # Upper Confidence Bound
    THOMPSONSAMPLING = "ThompsonSampling"
    SAMPLING = (UCB, THOMPSONSAMPLING)
    SOM = "SOM"
    RBM = "RBM"
    NONSUPERVISED = (SOM, RBM)
    ANN = "ann"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    NEURAL = (ANN, CNN, RNN)

    # graph constant
    TRAINING_SET = "Training Set"
    TEST_SET = "Test Set"
    TITLE = "title"
    XLABEL = "xlabel"
    YLABEL = "ylabel"
    SMOOTH = "smooth"
    SMOOTH_VALUE = 0.01
    CLASSIFICATION_REPORT = "Classification Report"
    CONFUSION_MATRIX = "Confusion Matrix"
    PREDICTION_ERROR = "Prediction Error"
    ROC_AUC = "ROC AUC"
    INTERCLUSTER_DISTANCE = "Intercluster Distance"
    SILHOUETTE = "silhouette"

    # delimeter constant
    DIR_DELIM = "\\"
    EXT_DELIM = "."
    PIPE = '|'

    # datetime constant
    DT_FMT = "%Y%m%d"
    DTTM_FMT = "%Y%m%d_%H%M%S"

    # extension constant
    CSV = "csv"
    TSV = "tsv"
    KERAS = "keras" # as a tag for tf.keras.datasets
    GENERATED = "auto-generated"
    XLSX = "xlsx"
    GRAPHIC_EXT = "png"

    # data constant
    MAX_VOCAB_SIZE = 20000