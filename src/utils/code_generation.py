from datetime import datetime
from re import sub

CODE_GEN_HEADER = (
    f"# THIS FILE WAS AUTOMATICALLY GENERATED AT {datetime.now().isoformat()}\n"
    "#####################################################################\n\n"
)


def get_code_gen_header() -> str:
    return CODE_GEN_HEADER


def sanitize_attribute(attribute: str) -> str:
    """
    Sanitize an attribute name for use in a Python class.
    """
    char_map = {
        " ": "_",
        "-": "_",
        r"\.": "_dot_",
        ",": "_comma_",
        r"\(": "_leftP_",
        r"\)": "_rightP_",
        "<": "_lt_",
        ">": "_gt_",
        "/": "_slash_",
        r"\b\d": "_num_",
    }

    for char, replacement in char_map.items():
        attribute = sub(char, replacement, attribute)

    return attribute
