LABEL_MAP = {
    'Negative': 0, 'Neutral': 1, 'Positive': 2
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def encoder(text) -> int:
    return LABEL_MAP.get(text, 7)


def decoder(number: int) -> str:
    return REVERSE_LABEL_MAP.get(number, "Neutral")
