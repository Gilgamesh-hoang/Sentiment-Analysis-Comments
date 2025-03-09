LABEL_MAP = {
    "Enjoyment": 0,
    "Disgust": 1,
    "Sadness": 2,
    "Anger": 3,
    "Surprise": 4,
    "Fear": 5,
    "Happy": 6,
    "Other": 7
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def encoder(text) -> int:
    return LABEL_MAP.get(text, 7)


def decoder(number: int) -> str:
    return REVERSE_LABEL_MAP.get(number, "Other")
