# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import os
import io
import urllib.parse
import datetime
import requests
import json
import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage


# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# from .helpers.openai_info import get_openai_callback
from .helpers.openai import get_openai_token_cost_for_model
import boto3
from botocore.exceptions import ClientError
from typing import Any, Text, Dict, List
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from rasa_sdk import Action, FormValidationAction, Tracker
from rasa_sdk.events import SlotSet, EventType, AllSlotsReset, UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

load_dotenv()

ALLOWED_FILE_FORMATS = ["csv", "xls", "xlsx", "pdf"]

aggregate_titles = ["total", "totales"]

# Create a new client and connect to the server
mongo_client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi("1"))

# Set OpenAI API key
client = OpenAI()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))


def get_completion(
    conversation_id: str,
    rasa_action: str,
    prompt: str,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4"),
) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model=model,  # this is the model that the API will use to generate the response
        messages=[
            {"role": "user", "content": prompt}
        ],  # this is the prompt that the model will complete
        temperature=0.5,  # this is the degree of randomness of the model's output
        max_tokens=2000,  # this is the maximum number of tokens that the model can generate
        top_p=1,  # this is the probability that the model will generate a token that is in the top p tokens
        frequency_penalty=0,  # this is the degree to which the model will avoid repeating the same line
        presence_penalty=0,  # this is the degree to which the model will avoid generating offensive language
    )

    # Log completion to MongoDB
    try:
        mongo_client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
        db = mongo_client.get_database("logs")
        collection = db.get_collection("completions")
        document = {
            "created": response.created,
            "conversation_id": conversation_id,
            "model": response.model,
            "rasa_action": rasa_action,
            "prompt": prompt,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": get_openai_token_cost_for_model(
                    model_name=model, num_tokens=response.usage.total_tokens
                ),
            },
        }
        collection.insert_one(document)
    except Exception as e:
        print(e)

    return response.choices[0].message.content


# def get_completion_with_pandasai(
#     conversation_id: str,
#     rasa_action: str,
#     prompt: str,
#     dataframe: pd.DataFrame,
#     model: str = os.environ.get("OPENAI_MODEL", "gpt-4"),
# ) -> str:
#     sdf = SmartDataframe(
#         df=dataframe,
#         config={
#             "llm": llm,
#             "custom_instructions": "The query will be made in Spanish and the results will be returned in Spanish.",
#         },
#     )
#     sdf.chat(prompt)
#     with get_openai_callback() as cb:
#         response = sdf.chat(prompt)
#         print(response)
#         print(cb)

#     # Log completion to MongoDB
#     try:
#         client.admin.command("ping")
#         print("Pinged your deployment. You successfully connected to MongoDB!")
#         db = client.get_database("logs")
#         collection = db.get_collection("completions")
#         document = {
#             "created": datetime.datetime.now(),
#             "conversation_id": conversation_id,
#             "model": model,
#             "rasa_action": rasa_action,
#             "prompt": prompt,
#             "usage": {
#                 "prompt_tokens": cb.prompt_tokens,
#                 "completion_tokens": cb.completion_tokens,
#                 "total_tokens": cb.total_tokens,
#                 "cost": cb.total_cost,
#             },
#         }
#         collection.insert_one(document)
#     except Exception as e:
#         print(e)

#     return response


def clear_menu(conversation_id: str, message_id: str, dispatcher: CollectingDispatcher):
    # Define the Telegram API URL and the parameters for the deleteMessage method
    url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/deleteMessage"
    params = {"chat_id": conversation_id, "message_id": message_id}

    # Send the request to the Telegram API
    response = requests.post(url, params=params)

    # Check the response
    if response.status_code == 200:
        dispatcher.utter_message(text="Menu removed successfully.")
    else:
        dispatcher.utter_message(text="Failed to remove menu.")


class ValidateDataSearchTermsForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_data_search_terms_form"

    def validate_data_search_terms(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `data_search_terms` value."""

        url_search = f"{os.getenv('DKAN_API')}/search?fulltext={urllib.parse.quote(slot_value.lower())}"
        response_search = requests.request("GET", url_search, headers={}, data={})
        response_search_json = response_search.json()
        num_items = response_search_json["total"]

        print("url_search in validation:", url_search)

        if num_items == "0":
            dispatcher.utter_message(text="Lo siento, no tengo datos sobre ese tema.")
            return {"data_search_terms": None}
        dispatcher.utter_message(text=f"OK! Quieres datos sobre {slot_value}.")
        return {"data_search_terms": slot_value}


class ValidateDataDownloadForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_data_download_form"

    def validate_data_file_format(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `data_file_format` value."""

        print("DATA_FILE_FORMAT Validation")

        data_meta = tracker.get_slot("data_meta")

        allowed_file_formats_dynamic = []

        for distribution in data_meta["distribution"]:
            allowed_file_formats_dynamic.append(distribution["format"].lower())

        if slot_value.lower() not in allowed_file_formats_dynamic:
            dispatcher.utter_message(text="El formato no es valido.")
            return {"data_file_format": None}
        dispatcher.utter_message(
            text=f"¡Ok! Quieres descargar archivos en formato {slot_value}."
        )
        return {"data_file_format": slot_value}


class AskForDataFileFormat(Action):
    def name(self) -> Text:
        return "action_ask_data_file_format"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher)

        data_meta = tracker.get_slot("data_meta")

        buttons = []

        for distribution in data_meta["distribution"]:
            buttons.append(
                [
                    {
                        "text": distribution["format"].upper(),
                        "callback_data": distribution["format"].lower(),
                    }
                ]
            )

        keyboard = json.dumps({"inline_keyboard": buttons})

        request_url = (
            f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/sendMessage"
        )
        params = {
            "chat_id": conversation_id,
            "text": "¿En qué formato quieres los datos?",
            "reply_markup": keyboard,
        }

        try:
            response = requests.post(
                request_url,
                params=params,
            )
        except Exception as e:
            print(e)

        if response.ok:
            print("Menu rendered.")
            print(response.json())
            return [SlotSet("menu_message_id", response.json()["result"]["message_id"])]
        else:
            print("Error rendering menu.")
            print(response.json())
            dispatcher.utter_message(text="Error al mostrar el menu.")
            return []


class ActionSearchData(Action):
    def name(self) -> Text:
        return "action_search_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id

        # Search dataset with search terms
        search_terms = tracker.get_slot("data_search_terms") or ""
        # print(search_terms)
        url_search = f"{os.getenv('DKAN_API')}/search?fulltext={urllib.parse.quote(search_terms)}"
        # print("url_search in action:", url_search)
        payload = {}
        headers = {}
        response_search = requests.request(
            "GET", url_search, headers=headers, data=payload
        )
        response_search_json = response_search.json()
        # print(response_search_json)

        dataset_keys = list(response_search_json["results"].keys())
        dataset_id = response_search_json["results"][dataset_keys[0]]["identifier"]
        # print(dataset_id)

        datastore_index = 0
        dataset_distributions = response_search_json["results"][dataset_keys[0]][
            "distribution"
        ]
        # print(dataset_distributions)

        # Find datastore index with csv format
        for idx, item in enumerate(dataset_distributions):
            if item["format"] == "csv":
                datastore_index = idx

        # print("Datastore index:", datastore_index)

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
        dispatcher.utter_message(text=f"He encontrado el dataset con id: {dataset_id}.")
        # dispatcher.utter_message(json_message=response.json())

        return [
            SlotSet("data", response_datastore.json()),
            SlotSet("data_meta", response_meta.json()),
        ]


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
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher)

        dispatcher.utter_message(text="Generando gráfico...")

        data = tracker.get_slot("data")
        data_meta = tracker.get_slot("data_meta")
        # print(data)
        # print(data_meta)

        # Create image
        df = pd.json_normalize(data, record_path=["results"])
        # Transform suitable data to numeric or keep as string
        df = df.apply(pd.to_numeric, errors="coerce").fillna(df)

        # df_csv = pd.read_csv(
        #     "http://datos.cadiz.local.ddev.site/sites/default/files/uploaded_resources/residuos-recogidos-por-servicios-municipales-v1.0.0.csv",
        #     delimiter=",",
        # )
        # print("CSV loaded")
        # print("DATAFRAME:", df)
        # print("DATAFRAME:", df.shape)
        # print("DATAFRAME:", df.info())
        # print("CSV:", df_csv)
        # print("CSV:", df_csv.shape)
        # print("CSV:", df_csv.info())

        # plt.style.use("seaborn")
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

            # print(property)
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
            s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_REGION"),
                endpoint_url=f"https://s3.{os.getenv('AWS_REGION')}.amazonaws.com",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except ClientError as ce:
            print("error", ce)
        finally:
            try:
                # Save image to S3
                s3_client.put_object(
                    Body=img_data,
                    ContentType="image/png",
                    Bucket=os.getenv("S3_BUCKET_NAME"),
                    Key=f"{conversation_id}.png",
                    Expires=datetime.datetime.now() + datetime.timedelta(days=1),
                )

                # Get a presigned url to avoid public access
                presigned_image_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": f"{os.getenv('S3_BUCKET_NAME')}",
                        "Key": f"{conversation_id}.png",
                    },
                    ExpiresIn=3600,
                )
                dispatcher.utter_message(
                    image=presigned_image_url,
                )
            except ClientError as ce:
                print("error", ce)
            finally:
                s3_client.close()
                # Send image

                # # Ask if user wants to see a plot
                # dispatcher.utter_button_message(
                #     text="¿Quieres que explique los datos?",
                #     buttons=[
                #         {"title": "Sí", "payload": "Explícame los datos"},
                #         {"title": "No", "payload": "/no"},
                #     ],
                # )

        return [SlotSet("menu_message_id", None)]


class ActionDownloadData(Action):
    def name(self) -> Text:
        return "action_download_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # theres a bug with rasa use of aiogram in utter_send_file for some file formats,
        # pdf works ok but csv and xlsx don't, so we are using the telegram api directly
        # aiogram.utils.exceptions.WrongFileIdentifier: Wrong file identifier/http url specified
        # once the bug is fixed we can use the dispatcher.utter_message
        # dispatcher.utter_message(response="utter_send_file", file_url=document_url)

        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher)

        data_meta = tracker.get_slot("data_meta")
        data_file_format = tracker.get_slot("data_file_format")
        data_distributions = data_meta["distribution"]

        print("data distributions type", type(data_distributions))
        print("FILE FORMAT:", data_file_format)

        document_index = None
        for i, obj in enumerate(data_distributions):
            print(i, obj["format"])
            if obj["format"] == data_file_format.lower():
                document_index = i
                break

        print("document_index:", document_index)

        request_url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/sendDocument"
        document_url = data_meta["distribution"][document_index]["downloadURL"]
        filename = os.path.basename(document_url)
        response_document = requests.get(document_url, verify=True)

        if response_document.ok:
            print("CSV file downloaded successfully.")
        else:
            print("Failed to download the CSV file.")

        files = {
            "document": (filename, response_document.content),
        }
        params = {"chat_id": conversation_id, "filename": filename}
        response = requests.post(request_url, files=files, params=params)

        if response.ok:
            print("CSV file sent successfully.")
        else:
            print("Failed to send the CSV file.")
            dispatcher.utter_message(text="Error enviando el archivo.")

        return [SlotSet("menu_message_id", None)]


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
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher)

        data = tracker.get_slot("data")
        print(data["results"])
        prompt = f"""
        Summarize the dataset in json delimited by triple backticks into a single sentence in spanish.
        ```{data["results"]}```
        """
        try:
            response = get_completion(
                conversation_id=conversation_id,
                rasa_action="action_explain_data",
                prompt=prompt,
            )
            print(response)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(
                text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
            )
        # dispatcher.utter_message(text="ChatGPT en modo debug (code: 001).")

        return [SlotSet("menu_message_id", None)]


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
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher)

        data = tracker.get_slot("data")
        # prompt = f"Summarize the data in spanish. Include statistics such as the number of rows and columns, the mean, median, mode, and standard deviation of each column, and the correlation between columns."

        prompt = f"""
        Summarize the data in json delimited by triple backticks in spanish. Include statistics such as the number of rows and columns, the mean, median, mode, and standard deviation of each column, and the correlation between columns.
        ```{data["results"]}```
        """

        # Create pandas dataframe with json string
        # df = pd.read_json(json.dumps(data["results"]))

        # print(df)
        # print(df.to_markdown())

        try:
            response = get_completion(
                conversation_id=conversation_id,
                rasa_action="action_statistics_data",
                prompt=prompt,
            )
            print(response)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(
                text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
            )
        # dispatcher.utter_message(text="ChatGPT en modo debug (code: 002).")

        return [SlotSet("menu_message_id", None)]


class ActionCustomQueryData(Action):
    def name(self) -> Text:
        return "action_custom_query_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher)

        data = tracker.get_slot("data")
        query = tracker.get_slot("data_custom_query")

        prompt = f"""
        I need you to answer a question in spanish about the data in json delimited by triple backticks:
        The question is: {query}
        
        ```{data["results"]}```
        """

        try:
            response = get_completion(
                conversation_id=conversation_id,
                rasa_action="action_custom_query_data",
                prompt=prompt,
            )
            print(response)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(
                text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
            )

        # Create pandas dataframe with json string
        # df = pd.read_json(json.dumps(data["results"]))

        # print(df)
        # print(df.to_markdown())

        # try:
        #     response = get_completion_with_pandasai(
        #         conversation_id=conversation_id,
        #         rasa_action="action_statistics_data",
        #         prompt=prompt,
        #         dataframe=df,
        #     )
        #     print(response)
        #     dispatcher.utter_message(text=response)
        # except Exception as e:
        #     dispatcher.utter_message(
        #         text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
        #     )
        # dispatcher.utter_message(text=f"ChatGPT en modo debug (code: 003): {prompt}")

        return [SlotSet("data_custom_query", None), SlotSet("menu_message_id", None)]


class ActionEmptySlots(Action):
    def name(self) -> Text:
        return "action_empty_slots"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        return [AllSlotsReset()]


class ActionShowMenu(Action):
    def name(self) -> Text:
        return "action_show_menu"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        conversation_id = tracker.sender_id

        buttons = [
            [
                {
                    "text": "Mostar gráfico",
                    "callback_data": "/plot_data",
                }
            ],
            [
                {
                    "text": "Descargar archivo",
                    "callback_data": "/download_data",
                }
            ],
            [
                {
                    "text": "Explicar datos",
                    "callback_data": "/explain_data",
                }
            ],
            [
                {
                    "text": "Mostrar datos estadísticos",
                    "callback_data": "/statistics_data",
                }
            ],
            [
                {
                    "text": "Consulta personalizada",
                    "callback_data": "/custom_query_data",
                }
            ],
            [
                {
                    "text": "Reiniciar",
                    "callback_data": "/stop",
                }
            ],
        ]

        keyboard = json.dumps({"inline_keyboard": buttons})

        request_url = (
            f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/sendMessage"
        )
        params = {
            "chat_id": conversation_id,
            "text": "¿Quieres saber algo sobre los datos?",
            "reply_markup": keyboard,
        }

        try:
            response = requests.post(
                request_url,
                params=params,
            )
        except Exception as e:
            print(e)

        if response.ok:
            print("Menu rendered.")
            print(response.json())
            return [SlotSet("menu_message_id", response.json()["result"]["message_id"])]
        else:
            print("Error rendering menu.")
            print(response.json())
            dispatcher.utter_message(text="Error al mostrar el menu.")
            return []


# class ActionCustomFallback(Action):
#     def name(self) -> Text:
#         return "action_custom_fallback"

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         dispatcher.utter_message(template="utter_unexpected")

#         return [UserUtteranceReverted()]
