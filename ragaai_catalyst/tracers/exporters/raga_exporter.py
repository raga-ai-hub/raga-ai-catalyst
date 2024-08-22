import os
import json
import asyncio
import aiohttp
import logging
from tqdm import tqdm
import requests
from ...ragaai_catalyst import RagaAICatalyst
import shutil

logger = logging.getLogger(__name__)

get_token = RagaAICatalyst.get_token


class RagaExporter:
    BASE_URL = None
    SCHEMA_MAPPING = {
        "trace_id": "traceId",
        "trace_uri": "traceUri",
        "prompt": "prompt",
        "response": "response",
        "context": "context",
        "llm_model": "pipeline",
        "recorded_on": "metadata",
        "embed_model": "pipeline",
        "log_source": "metadata",
        "vector_store": "pipeline",
    }
    TIMEOUT = 10

    def __init__(self, project_name):
        """
        Initializes a new instance of the RagaExporter class.

        Args:
            project_name (str): The name of the project.

        Raises:
            ValueError: If the environment variables RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY are not set.
            Exception: If the schema check fails or the schema creation fails.
        """
        self.project_name = project_name
        RagaExporter.BASE_URL = (
            os.getenv("RAGAAI_CATALYST_BASE_URL")
            if os.getenv("RAGAAI_CATALYST_BASE_URL")
            else "https://llm-platform.dev4.ragaai.ai/api"
        )
        self.access_key = os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
        self.secret_key = os.getenv("RAGAAI_CATALYST_SECRET_KEY")
        self.max_urls = 20
        if not self.access_key or not self.secret_key:
            raise ValueError(
                "RAGAAI_CATALYST_ACCESS_KEY and RAGAAI_CATALYST_SECRET_KEY environment variables must be set"
            )
        if not os.getenv("RAGAAI_CATALYST_TOKEN"):
            get_token()
        status_code = self._check_schema()
        if status_code == 404:
            create_status_code = self._create_schema()
            if create_status_code != 200:
                raise Exception(
                    "Failed to create schema. Please consider raising an issue."
                )
        elif status_code != 200:
            raise Exception("Failed to check schema. Please consider raising an issue.")

    def _check_schema(self):
        """
        Checks if the schema for the project exists.

        This function makes a GET request to the RagaExporter.BASE_URL endpoint to check if the schema for the project exists.
        It uses the project name to construct the URL.

        Returns:
            int: The status code of the response. If the response status code is 200, it means the schema exists.
                 If the response status code is 401, it means the token is invalid and a new token is fetched and set in the environment.
                 If the response status code is not 200, it means the schema does not exist.

        Raises:
            None
        """

        def make_request():
            headers = {
                "authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            response = requests.get(
                f"{RagaExporter.BASE_URL}/v1/llm/master-dataset/schema/{self.project_name}",
                headers=headers,
                timeout=RagaExporter.TIMEOUT,
            )
            return response

        response = make_request()
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return response.status_code
        return response.status_code

    def _create_schema(self):
        """
        Creates a schema for the project by making a POST request to the RagaExporter.BASE_URL endpoint.

        This function makes a POST request to the RagaExporter.BASE_URL endpoint to create a schema for the project.
        It uses the project name and the schema mapping defined in RagaExporter.SCHEMA_MAPPING to construct the JSON data.
        The request includes the project name, schema mapping, and a trace folder URL set to None.

        Parameters:
            self (RagaExporter): The instance of the RagaExporter class.

        Returns:
            int: The status code of the response. If the response status code is 200, it means the schema was created successfully.
                 If the response status code is 401, it means the token is invalid and a new token is fetched and set in the environment.
                 If the response status code is not 200, it means the schema creation failed.

        Raises:
            None
        """

        def make_request():
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            json_data = {
                "projectName": self.project_name,
                "schemaMapping": RagaExporter.SCHEMA_MAPPING,
                "traceFolderUrl": None,
            }
            response = requests.post(
                f"{RagaExporter.BASE_URL}/v1/llm/master-dataset",
                headers=headers,
                json=json_data,
                timeout=RagaExporter.TIMEOUT,
            )
            return response

        response = make_request()
        if response.status_code == 401:
            get_token()  # Fetch a new token and set it in the environment
            response = make_request()  # Retry the request
        if response.status_code != 200:
            return response.status_code
        return response.status_code

    async def response_checker_async(self, response, context=""):
        logger.debug(f"Function: {context} - Response: {response}")
        status_code = response.status
        return status_code

    async def get_presigned_url(self, session, num_files):
        """
        Asynchronously retrieves a presigned URL from the RagaExporter API.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            num_files (int): The number of files to be uploaded.

        Returns:
            dict: The JSON response containing the presigned URL.

        Raises:
            aiohttp.ClientError: If the request fails.

        """

        async def make_request():
            json_data = {
                "projectName": self.project_name,
                "numFiles": num_files,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "X-Project-Name": self.project_name,
            }
            async with session.get(
                f"{RagaExporter.BASE_URL}/v1/llm/presigned-url",
                headers=headers,
                json=json_data,
                timeout=RagaExporter.TIMEOUT,
            ) as response:

                # print(json_response)
                json_data = await response.json()
                return response, json_data

        response, json_data = await make_request()
        await self.response_checker_async(response, "RagaExporter.get_presigned_url")
        if response.status == 401:
            await get_token()  # Fetch a new token and set it in the environment
            response, json_data = await make_request()  # Retry the request

        if response.status != 200:
            return {"status": response.status, "message": "Failed to get presigned URL"}

        return json_data

    async def stream_trace(self, session, trace_uri):
        """
        Asynchronously streams a trace to the RagaExporter API.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            trace_uri (str): The URI of the trace to stream.

        Returns:
            int: The status code of the response.

        Raises:
            aiohttp.ClientError: If the request fails.

        """

        async def make_request():

            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "Content-Type": "application/json",
                "X-Project-Name": self.project_name,
            }

            json_data = {
                "projectName": self.project_name,
                "traceUri": trace_uri,
            }

            async with session.post(
                f"{RagaExporter.BASE_URL}/v1/llm/insert/trace",
                headers=headers,
                json=json_data,
                timeout=RagaExporter.TIMEOUT,
            ) as response:

                status = response.status
                return response, status

        response, status = await make_request()
        await self.response_checker_async(response, "RagaExporter.upload_file")
        if response.status == 401:
            await get_token()  # Fetch a new token and set it in the environment
            response, status = await make_request()  # Retry the request

        if response.status != 200:
            return response.status

        return response.status


    async def check_and_upload_files(self, session, file_paths):
        """
        Checks if there are files to upload, gets presigned URLs, uploads files, and streams them if successful.

        Args:
            self: The object instance.
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            file_paths (list): List of file paths to upload.

        Returns:
            str: The status of the upload process.
        """ """
        Asynchronously uploads a file using the given session, url, and file path.

        Args:
            self: The RagaExporter instance.
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            url (str): The URL to upload the file to.
            file_path (str): The path to the file to upload.

        Returns:
            int: The status code of the response.
        """
        # Check if there are no files to upload
        if len(file_paths) == 0:
            logger.info("No files to be uploaded.")
            return None

        # Ensure a required environment token is available; if not, attempt to obtain it.
        if os.getenv("RAGAAI_CATALYST_TOKEN") is None:
            await get_token()
            if os.getenv("RAGAAI_CATALYST_TOKEN") is None:
                print("Failed to obtain token.")
                return None

        # Initialize lists for URLs and tasks
        presigned_urls = []
        trace_folder_urls = []
        num_files = len(file_paths)

        # Fetch presigned URLs
        for i in range((num_files - 1) // self.max_urls + 1):
            batch_size = min(self.max_urls, num_files - i * self.max_urls)
            presigned_url_response = await self.get_presigned_url(session, batch_size)
            if presigned_url_response.get("success") == True:
                data = presigned_url_response.get("data", {})
                presigned_urls.extend(data.get("presignedUrls", []))
                trace_folder_urls.append(data.get("traceFolderUrl"))
            else:
                logger.error(f"Failed to get presigned URLs for batch {i + 1}")
                return None

        if not presigned_urls:
            logger.error("Failed to get any presigned URLs.")
            return None

        # Upload and stream files
        for file_path, presigned_url in tqdm(zip(file_paths, presigned_urls), total=num_files, desc="Uploading traces"):
            if not os.path.isfile(file_path):
                logger.warning(f"The file '{file_path}' does not exist.")
                continue

            try:
                upload_status = await self.upload_file(session, presigned_url, file_path)
                if upload_status in (200, 201):
                    logger.debug(f"File '{os.path.basename(file_path)}' uploaded successfully.")
                    stream_status = await self.stream_trace(session, trace_uri=presigned_url)
                    if stream_status in (200, 201):
                        logger.debug(f"File '{os.path.basename(file_path)}' streamed successfully.")
                        self._backup_file(file_path)
                    else:
                        logger.error(f"Failed to stream the file '{os.path.basename(file_path)}'.")
                else:
                    logger.error(f"Failed to upload the file '{os.path.basename(file_path)}'.")
            except Exception as e:
                logger.error(f"Error processing file '{os.path.basename(file_path)}': {str(e)}")

        return "Upload completed"

    def _backup_file(self, file_path):
        backup_dir = os.path.join(os.path.dirname(file_path), "backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, f"{os.path.basename(file_path).split('.')[0]}_backup.json")
        shutil.move(file_path, backup_file)

    async def upload_file(self, session, url, file_path):
        headers = {"Content-Type": "application/json"}
        if "blob.core.windows.net" in url:
            headers["x-ms-blob-type"] = "BlockBlob"

        logger.debug(f"Uploading file: {file_path} with URL: {url}")

        async with aiohttp.ClientTimeout(total=self.TIMEOUT):
            with open(file_path, 'rb') as f:
                data = f.read()
            async with session.put(url, headers=headers, data=data) as response:
                status = response.status
                if status not in (200, 201):
                    logger.error(f"Upload failed with status {status}")
                return status


    async def tracer_stopsession(self, file_names):
        """
        Asynchronously stops the tracing session, checks for RAGAAI_CATALYST_TOKEN, and uploads files if the token is present.

        Parameters:
            self: The current instance of the class.
            file_names: A list of file names to be uploaded.

        Returns:
            None
        """

        async with aiohttp.ClientSession() as session:
            if os.getenv("RAGAAI_CATALYST_TOKEN"):
                logger.info("Token obtained successfully.")
                result = await self.check_and_upload_files(session, file_paths=file_names)
                logger.info(f"Upload process result: {result}")
            else:
                logger.error("Failed to obtain token.")
