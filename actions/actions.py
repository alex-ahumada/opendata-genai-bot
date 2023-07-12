# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

import os
import io
import datetime
import requests
import boto3
from typing import Any, Text, Dict, List
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

load_dotenv()


class ActionFetchData(Action):
    def name(self) -> Text:
        return "action_fetch_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        # Fetch dataset datastore
        url = "http://datos.cadiz.local.ddev.site/api/1/datastore/query/fae52fed-1e83-48c0-8b75-304ae43d758d/1?count=true&results=true&schema=true&keys=true&format=json"
        payload = {}
        headers = {}
        # response = requests.request("GET", url, headers=headers, data=payload)
        # print(response.text)

        # Send message as json
        # dispatcher.utter_message(text=response.text)
        # dispatcher.utter_message(json_message=response.json())

        # Create image
        plt.plot([1, 2, 3], [1, 4, 9])
        img_data = io.BytesIO()
        plt.savefig(img_data, format="png")
        img_data.seek(0)

        # Save image to S3
        print(os.getenv("S3_BUCKET_NAME"))
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KET_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        bucket = s3.Bucket(os.getenv("S3_BUCKET_NAME"))
        bucket.put_object(
            Body=img_data,
            ContentType="image/png",
            Key=f"{conversation_id}.png",
            Expires=datetime.datetime.now() + datetime.timedelta(days=1),
        )
        # Send image
        dispatcher.utter_message(
            image=f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{conversation_id}.png"
        )

        return []


class ActionShowPlot(Action):
    def name(self) -> Text:
        return "action_show_plot"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id

        return []


class ActionExplainData(Action):
    def name(self) -> Text:
        return "action_explain_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id

        return []
