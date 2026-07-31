# intent_parser.py

def parse_request(text):

    text = text.lower().strip()

    if "help" in text:
        return "Teacher's help"

    if "pencil" in text:
        return "Need a pencil"

    if "eraser" in text or "rubber" in text:
        return "Need an eraser"

    if "block" in text or "blocks" in text:
        return "Need a block"

    return "Unknown request"
