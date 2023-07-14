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
import openai
import boto3
from botocore.config import Config
from typing import Any, Text, Dict, List
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

load_dotenv()

my_config = Config(
    region_name=os.getenv("AWS_REGION"),
)


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,  # this is the model that the API will use to generate the response
        messages=messages,  # this is the prompt that the model will complete
        temperature=0.5,  # this is the degree of randomness of the model's output
        max_tokens=256,  # this is the maximum number of tokens that the model can generate
        top_p=1,  # this is the probability that the model will generate a token that is in the top p tokens
        frequency_penalty=0,  # this is the degree to which the model will avoid repeating the same line
        presence_penalty=0,  # this is the degree to which the model will avoid generating offensive language
    )
    return response.choices[0].message["content"]


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
        url = f"{os.getenv('DKAN_API')}/datastore/query/fae52fed-1e83-48c0-8b75-304ae43d758d/1?count=true&results=true&schema=true&keys=true&format=json"
        payload = {}
        headers = {}
        response = requests.request("GET", url, headers=headers, data=payload)

        # Send message as json
        dispatcher.utter_message(text=response.text)
        # dispatcher.utter_message(json_message=response.json())

        # Ask if user wants to see a plot
        dispatcher.utter_button_message(
            text="¿Quieres ver un gráfico?",
            buttons=[
                {"title": "Sí", "payload": "Quiero ver un gráfico"},
                {"title": "No", "payload": "/no"},
            ],
        )

        return [SlotSet("data", response.json())]


class ActionPlotData(Action):
    def name(self) -> Text:
        return "action_plot_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id

        data = tracker.get_slot("data")
        print(data)

        # Create image
        plt.plot([1, 2, 3], [1, 4, 9])
        img_data = io.BytesIO()
        plt.savefig(img_data, format="png")
        img_data.seek(0)

        # Create S3 client
        # boto3 currently has a bug with regions launched after 2019
        # this is fixed by setting the endpoint_url
        s3_client = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION"),
            endpoint_url=f"https://s3.{os.getenv('AWS_REGION')}.amazonaws.com",
            aws_access_key_id=os.getenv("AWS_ACCESS_KET_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        # Save image to S3
        # s3 = boto3.resource(
        #     "s3",
        #     aws_access_key_id=os.getenv("AWS_ACCESS_KET_ID"),
        #     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        # )
        # bucket = s3.Bucket(os.getenv("S3_BUCKET_NAME"))
        # bucket.put_object(
        #     Body=img_data,
        #     ContentType="image/png",
        #     Key=f"{conversation_id}.png",
        #     Expires=datetime.datetime.now() + datetime.timedelta(days=1),
        # )

        s3_client.put_object(
            Body=img_data,
            ContentType="image/png",
            Bucket=os.getenv("S3_BUCKET_NAME"),
            Key=f"{conversation_id}.png",
            # Expires=datetime.datetime.now() + datetime.timedelta,
        )

        # Get a presigned url to avoid public access
        presigned_image_url = s3_client.generate_presigned_url(
            "get_object",
            ExpiresIn=3600,
            Params={
                "Bucket": f"{os.getenv('S3_BUCKET_NAME')}",
                "Key": f"{conversation_id}.png",
            },
        )

        # Send image
        dispatcher.utter_message(
            image=presigned_image_url,
        )

        # Ask if user wants to see a plot
        dispatcher.utter_button_message(
            text="¿Quieres que explique los datos?",
            buttons=[
                {"title": "Sí", "payload": "Explícame los datos"},
                {"title": "No", "payload": "/no"},
            ],
        )

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

        # Set OpenAI API key
        print(os.getenv("OPENAI_API_KEY"))
        openai.api_key = os.getenv("OPENAI_API_KEY")

        data = tracker.get_slot("data")
        prompt = f"""
        Summarize the dataset in json delimited by triple backticks \ 
        into a single sentence in spanish.
        ```{data}```
        """

        response = get_completion(prompt)
        print(response)
        dispatcher.utter_message(text=response)

        return []


class ActionStatisticsData(Action):
    def name(self) -> Text:
        return "action_statistics_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id

        # Set OpenAI API key
        print(os.getenv("OPENAI_API_KEY"))
        openai.api_key = os.getenv("OPENAI_API_KEY")

        data = tracker.get_slot("data")
        prompt = f"""
        Summarize the dataset in json delimited by triple backticks \ 
        in spanish. Include statistics such as \
        the number of rows and columns, the mean, median, mode, and \
        standard deviation of each column, and the correlation between \
        columns. \n
        ```{data}```
        """

        response = get_completion(prompt)
        print(response)
        dispatcher.utter_message(text=response)

        return []
