"""
Request Card Mapping
"""

CARD_MAP = {
    0: "blocks",
    1: "pencil",
    2: "eraser",
    3: "teacher"
}


def get_request(marker_id):
    """
    Convert an ArUco marker ID into a classroom request.

    Args:
        marker_id (int): Detected ArUco marker ID.

    Returns:
        str: Requested item or service.
    """
    return CARD_MAP.get(marker_id)