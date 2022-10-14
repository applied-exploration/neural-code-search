from enum import Enum


root_path: str = "data"


class Const:
    root_path: str = root_path
    root_data_original: str = f"{root_path}/original"
    root_data_processed: str = f"{root_path}/processed"


class DFCols(Enum):
    processed_feature = "extracted_code"
    embedded_feature = "embedded_feature"
    unprocessed_feature = "answer_body"
