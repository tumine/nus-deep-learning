# intent_parser.py

def parse_request(text):

    text = text.lower().strip()

    if "help" in text:
        return "Teacher's help"

    if "more" in text:
        return "Need one more _____"

    return "Unknown request"