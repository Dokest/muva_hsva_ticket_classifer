class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def color_ok(text: str) -> str:
    return f"{Color.OKGREEN}{text}{Color.ENDC}"


def color_fail(text: str) -> str:
    return f"{Color.WARNING}{text}{Color.ENDC}"
