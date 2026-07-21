"""
request_manager.py

Convert classroom requests into delivery tasks.
"""

class RequestManager:

    def create_task(self, result):
        """
        Convert a detected ArUco request into a delivery task.

        Args:
            result (dict): Detection result returned by CardDetector.

        Returns:
            dict: Delivery task.
        """

        return {

            "type": "delivery",

            "item": result["request"],

            "marker_id": result["id"],

            "target": result["center"]

        }