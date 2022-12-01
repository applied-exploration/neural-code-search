import logging
import sys

import html2text
import pandas as pd

stdout_logger = None


def pretty_print_results(original_code: str, df: pd.DataFrame) -> None:
    print(original_code)
    print("=" * 6)
    for index, row in df.iterrows():
        print(html2text.html2text(row["question_body"]))
        print("-" * 6)
        print(html2text.html2text(row["answer_body"]))


def get_logger():
    global stdout_logger
    if stdout_logger is not None:
        return stdout_logger

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    stdout_logger = logger
    return logger


def identity(anything):
    return anything
