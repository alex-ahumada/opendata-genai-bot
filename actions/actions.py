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
import urllib.parse
import datetime
import requests
import pandas as pd
import numpy as np
import openai
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Any, Text, Dict, List
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

load_dotenv()

my_config = Config(
    region_name=os.getenv("AWS_REGION"),
)

aggregate_titles = ["total", "totales"]


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

        # Search dataset with search terms
        search_terms = tracker.get_slot("data_search_terms")
        print(search_terms)
        url_search = f"{os.getenv('DKAN_API')}/search?fulltext={urllib.parse.quote(search_terms)}&page=1&page-size=20"
        print(url_search)
        payload = {}
        headers = {}
        response_search = requests.request(
            "GET", url_search, headers=headers, data=payload
        )
        response_search_json = response_search.json()
        print(response_search_json)

        dataset_keys = list(response_search_json["results"].keys())
        dataset_id = response_search_json["results"][dataset_keys[0]]["identifier"]
        print(dataset_id)

        datastore_index = 0
        dataset_distributions = response_search_json["results"][dataset_keys[0]][
            "distribution"
        ]
        print(dataset_distributions)

        # Find datastore index with csv format
        for idx, item in enumerate(dataset_distributions):
            if item["format"] == "csv":
                datastore_index = idx

        print(datastore_index)

        # Fetch dataset metadata
        url_meta = (
            f"{os.getenv('DKAN_API')}/metastore/schemas/dataset/items/{dataset_id}"
        )
        response_meta = requests.request("GET", url_meta, headers=headers, data=payload)

        # Fetch dataset datastore
        url_datastore = f"{os.getenv('DKAN_API')}/datastore/query/{dataset_id}/{datastore_index}?count=true&results=true&schema=true&keys=true&format=json"
        response_datastore = requests.request(
            "GET", url_datastore, headers=headers, data=payload
        )

        # Send message as json
        dispatcher.utter_message(text=f"El dataset {dataset_id} han sido encontrados.")
        # dispatcher.utter_message(json_message=response.json())

        # Ask if user wants to see a plot
        dispatcher.utter_button_message(
            text="¿Quieres ver un gráfico?",
            buttons=[
                {"title": "Sí", "payload": "Quiero ver un gráfico"},
                {"title": "No", "payload": "/no"},
            ],
        )

        return [
            SlotSet("data", response_datastore.json()),
            SlotSet("data_meta", response_meta.json()),
        ]


class ActionDownloadData(Action):
    def name(self) -> Text:
        return "action_download_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        return []


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
        data_meta = tracker.get_slot("data_meta")
        # print(data)
        print(data_meta)

        # Create image
        df = pd.json_normalize(data, record_path=["results"])
        # Transform suitable data to numeric or keep as string
        df = df.apply(pd.to_numeric, errors="coerce").fillna(df)
        # df = pd.read_csv(
        #     "https://datos.cadiz.local.ddev.site/sites/default/files/uploaded_resources/residuos-recogidos-por-servicios-municipales-v1.0.0.xlsx",
        #     # "actions/data/residuos-recogidos-por-servicios-municipales-v1.1.2.csv",
        #     # encoding="utf-8", encoding_errors="ignore",
        #     delimiter=",",
        #     # encoding="unicode_escape",
        # )
        # print("CSV loaded")
        # print(df)
        # print(df.shape)
        # print(df.info())

        plt.style.use("seaborn")
        plt.figure(figsize=(10, 5))
        # plt.title(data["query"]["resources"][0]["id"])
        plt.title(data_meta["title"])
        plt.xlabel(data["query"]["properties"][0])

        for property in data["query"]["properties"]:
            # skip first property
            if property == data["query"]["properties"][0]:
                continue
            # remove unwanted properties
            if property in aggregate_titles:
                continue

            print(property)
            # df[property] = pd.to_numeric(df[property], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[property]):
                plt.plot(
                    df[data["query"]["properties"][0]],
                    df[property],
                    marker=".",
                    markersize=10,
                    label=data["schema"][data["query"]["resources"][0]["id"]]["fields"][
                        property
                    ]["description"],
                )
        plt.legend(loc=(1.02, 0), borderaxespad=0, fontsize=12)
        plt.tight_layout()

        img_data = io.BytesIO()
        plt.savefig(img_data, format="png", dpi=72)
        img_data.seek(0)

        # Create S3 client
        # boto3 currently has a bug with regions launched after 2019
        # this is fixed by setting the endpoint_url in boto3.client
        # https://github.com/boto/boto3/issues/2864
        try:
            # We use boto3.client instead of boto3.resource because the bug
            # is not fixed in boto3.resource
            # s3 = boto3.resource(
            #     "s3",
            #     aws_access_key_id=os.getenv("AWS_ACCESS_KET_ID"),
            #     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            # )
            # bucket = s3.Bucket(os.getenv("S3_BUCKET_NAME"))
            s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_REGION"),
                endpoint_url=f"https://s3.{os.getenv('AWS_REGION')}.amazonaws.com",
                aws_access_key_id=os.getenv("AWS_ACCESS_KET_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except ClientError as ce:
            print("error", ce)
        finally:
            try:
                # Save image to S3
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
                    Expires=datetime.datetime.now() + datetime.timedelta(days=1),
                )

                # Get a presigned url to avoid public access
                # presigned_image_url = s3.meta.client.generate_presigned_url(
                #     "get_object",
                #     Params={
                #         "Bucket": f"{os.getenv('S3_BUCKET_NAME')}",
                #         "Key": f"{conversation_id}.png",
                #     },
                #     ExpiresIn=3600,
                # )
                presigned_image_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": f"{os.getenv('S3_BUCKET_NAME')}",
                        "Key": f"{conversation_id}.png",
                    },
                    ExpiresIn=3600,
                )
            except ClientError as ce:
                print("error", ce)
            finally:
                s3_client.close()
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

        # response = get_completion(prompt)
        # print(response)
        # dispatcher.utter_message(text=response)

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
