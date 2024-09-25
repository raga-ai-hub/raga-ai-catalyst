import os
import requests
import pandas as pd
import io
from .ragaai_catalyst import RagaAICatalyst
import logging
import pdb

logger = logging.getLogger(__name__)

class Evaluation:

    def __init__(self, project_name, dataset_name):
        self.project_name = project_name
        self.dataset_name = dataset_name
        self.base_url = f"{RagaAICatalyst.BASE_URL}"
        self.timeout = 10
        self.jobId = None

        try:
            response = requests.get(
                f"{self.base_url}/v2/llm/projects",
                headers={
                    "Authorization": f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}',
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Projects list retrieved successfully")

            self.project_id = [
                project["id"] for project in response.json()["data"]["content"] if project["name"] == project_name
            ][0]


            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Id": str(self.project_id),
            }
            json_data = {"size": 12, "page": "0", "projectId": str(self.project_id), "search": ""}
            response = requests.post(
                f"{self.base_url}/v2/llm/dataset",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )
            
            datasets_content = response.json()["data"]["content"]
            self.dataset_id = [dataset["id"] for dataset in datasets_content if dataset["name"]==dataset_name][0]


        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    
    def list_metrics(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{self.base_url}/v1/llm/llm-metrics', 
                headers=headers)
            response.raise_for_status()
            metric_names = [metric["name"] for metric in response.json()["data"]["metrics"]]
            return metric_names
        except:
            pass

    def get_mapping(self, metric_name, metrics_schema):
        mapping = []
        for schema in metrics_schema:
            if schema["name"]==metric_name:
                requiredFields = schema["config"]["requiredFields"]
                for field in requiredFields:
                    schemaName = field["name"]
                    variableName = schemaName
                    mapping.append({"schemaName": schemaName, "variableName": variableName})
        return mapping

    def _get_metricParams(self):
        return {
                "metricSpec": {
                    "name": "metric_to_evaluate",
                    "config": {
                        "model": "null",
                        "params": {
                            "model": {
                                "value": "gpt-4o"
                            },
                            "threshold": {
                                "gte": 0.5
                            }
                        },
                        "mappings": "mappings"
                    },
                    "displayName": "displayName"
                },
                "rowFilterList": []
            }
    
    def get_metrics_schema_response(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        try:
            response = requests.get(
                f'{self.base_url}/v1/llm/llm-metrics', 
                headers=headers)
            response.raise_for_status()
            metrics_schema = [metric for metric in response.json()["data"]["metrics"]]
            return metrics_schema
        except:
            pass


    def update_base_json(self, metrics):
        metric_schema_mapping = {"datasetId":self.dataset_id}
        metrics_schema_response = self.get_metrics_schema_response()
        metricParams = []
        for metric in metrics:
            base_json = self._get_metricParams()
            base_json["metricSpec"]["name"] = metric["name"]
            if metric["config"]["model"]:
                base_json["metricSpec"]["config"]["params"]["model"]["value"] = metric["config"]["model"]
            base_json["metricSpec"]["displayName"] = metric["column_name"]
            mappings = self.get_mapping(metric["name"], metrics_schema_response)
            base_json["metricSpec"]["config"]["mappings"] = mappings
            metricParams.append(base_json)
        metric_schema_mapping["metricParams"] = metricParams
        return metric_schema_mapping

    def _get_executed_metrics_list(self):
        headers = {
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        response = requests.get(
            f'{self.base_url}/v1/llm/filter?datasetId={str(self.dataset_id)}', 
            headers=headers
            )
        
        executed_metric_response = response.json()["data"]["filter"]
        executed_metric_list = [item["displayName"] for item in executed_metric_response]
        return executed_metric_list


    def add_metrics(self, metrics):
        executed_metric_list = self._get_executed_metrics_list()
        column_names = [metric["column_name"] for metric in metrics]
        for column_name in column_names:
            if column_name in executed_metric_list:
                raise ValueError(f"Column name '{column_name}' already exists.")

        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        metric_schema_mapping = self.update_base_json(metrics)
        try:
            response = requests.post(
                f'{self.base_url}/playground/metric-evaluation', 
                headers=headers, 
                json=metric_schema_mapping
                )
            response.raise_for_status()
            if response.json()["success"]:
                print(response.json()["message"])
                self.jobId = response.json()["data"]["jobId"]

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def get_status(self):
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
            'X-Project-Id': str(self.project_id),
        }
        data = {"jobId": self.jobId}
        try:
            response = requests.post(
                f'{self.base_url}/job/status', 
                headers=headers, 
                json=data)
            response.raise_for_status()
            print(response.json()["data"]["status"])
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")

    def get_results(self):


        def get_presignedUrl():
            headers = {
                'Content-Type': 'application/json',
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                'X-Project-Id': str(self.project_id),
                }
            
            data = {
                "fields": [
                    "*"
                ],
                "datasetId": str(self.dataset_id),
                "rowFilterList": [],
                "export": True
                }
                
            response = requests.post(
                f'{self.base_url}/v1/llm/docs', 
                headers=headers, 
                json=data)
            
            return response.json()

        def parse_response():
            response = get_presignedUrl()
            preSignedURL = response["data"]["preSignedURL"]
            response = requests.get(preSignedURL)
            return response.text
        
        response_text = parse_response()
        df = pd.read_csv(io.StringIO(response_text))

        column_list = df.columns.to_list()
        column_list = [col for col in column_list if not col.startswith('_')]
        column_list = [col for col in column_list if '.' not in col]
        return df[column_list]




