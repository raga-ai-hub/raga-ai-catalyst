import os
import json
import asyncio
import aiohttp
import logging
from tqdm import tqdm
import requests
from ...ragaai_catalyst import RagaAICatalyst
import shutil
import traceback

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
        "prompt_length": "promptLength",
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

    async def upload_file(self, session, url, file_path):
        """
        Asynchronously uploads a file using the given session, url, and file path.

        Args:
            self: The RagaExporter instance.
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            url (str): The URL to upload the file to.
            file_path (str): The path to the file to upload.

        Returns:
            int: The status code of the response.
        """

        async def make_request():

            headers = {
                "Content-Type": "application/json",
            }
            print(f"Uploading traces...")
            logger.debug(f"Uploading file:{file_path} with url {url}")
            with open(file_path) as f:
                data = f.read().replace("\n", "").replace("\r", "").encode()
            async with session.put(
                url, headers=headers, data=data, timeout=RagaExporter.TIMEOUT
            ) as response:

                # print(json_response)
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
    
    async def stream_trace(self, session, trace_uri):
        """
        Asynchronously streams a trace to the RagaExporter API.
        """
        async def make_request():
            headers = {
                "Authorization": f"Bearer {os.getenv('RAGAAI_CATALYST_TOKEN')}",
                "Content-Type": "application/json",
            }

            json_data = {
                "projectName": self.project_name,
                "traceUri": trace_uri,
            }

            logger.debug(f"Streaming trace with URI: {trace_uri}")
            logger.debug(f"Request headers: {headers}")
            logger.debug(f"Request payload: {json_data}")

            try:
                async with session.post(
                    f"{RagaExporter.BASE_URL}/v1/llm/insert/trace",
                    headers=headers,
                    json=json_data,
                    timeout=RagaExporter.TIMEOUT,
                ) as response:
                    status = response.status
                    logger.debug(f"Stream trace response status: {status}")
                    if status != 200:
                        response_text = await response.text()
                        logger.error(f"Stream trace error response: {response_text}")
                    return response, status
            except Exception as e:
                logger.error(f"Exception during stream trace: {str(e)}")
                logger.error(traceback.format_exc())
                return None, 500

        response, status = await make_request()
        await self.response_checker_async(response, "RagaExporter.stream_trace")
        if status == 401:
            logger.warning("Token expired. Fetching new token.")
            await get_token()
            response, status = await make_request()

        if status != 200:
            logger.error(f"Failed to stream trace. Status code: {status}")
            return status

        logger.info(f"Successfully streamed trace. Status code: {status}")
        return status




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
            print("No files to be uploaded.")
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
        tasks_json = []
        tasks_stream = []
        # Determine the number of files to process
        num_files = len(file_paths)

        # If number of files exceeds the maximum allowed URLs, fetch URLs in batches
        if num_files > self.max_urls:
            for i in range(
                (num_files // self.max_urls) + 1
            ):  # Correct integer division
                presigned_url_response = await self.get_presigned_url(
                    session, self.max_urls
                )
                if presigned_url_response.get("success") == True:
                    data = presigned_url_response.get("data", {})
                    presigned_urls += data.get("presignedUrls", [])
                    trace_folder_urls.append(data.get("traceFolderUrl", []))
        else:
            # Fetch URLs for all files if under the limit
            presigned_url_response = await self.get_presigned_url(session, num_files)
            if presigned_url_response.get("success") == True:
                data = presigned_url_response.get("data", {})
                presigned_urls += data.get("presignedUrls", [])
                trace_folder_urls.append(data.get("traceFolderUrl", []))

        # If URLs were successfully obtained, start the upload process
        
        if presigned_urls != []:
            for file_path, presigned_url in tqdm(
                zip(file_paths, presigned_urls), desc="Uploading traces"
            ):
                if not os.path.isfile(file_path):
                    logger.warning(f"The file '{file_path}' does not exist.")
                    continue

                logger.info(f"Attempting to upload file: {file_path}")
                upload_status = await self.upload_file(
                    session, presigned_url, file_path
                )
                if upload_status == 200:
                    logger.info(f"File '{os.path.basename(file_path)}' uploaded successfully.")
                    logger.info(f"Attempting to stream file: {file_path}")
                    stream_status = await self.stream_trace(
                        session, trace_uri=presigned_url
                    )
                    if stream_status == 200:
                        logger.info(f"File '{os.path.basename(file_path)}' streamed successfully.")
                        try:
                            shutil.move(
                                file_path,
                                os.path.join(
                                    os.path.dirname(file_path),
                                    "backup",
                                    os.path.basename(file_path).split(".")[0]
                                    + "_backup.json",
                                ),
                            )
                            logger.info(f"File '{os.path.basename(file_path)}' moved to backup.")
                        except Exception as e:
                            logger.error(f"Failed to move file to backup: {str(e)}")
                    else:
                        logger.error(f"Failed to stream the file '{os.path.basename(file_path)}'. Status code: {stream_status}")
                else:
                    logger.error(f"Failed to upload the file '{os.path.basename(file_path)}'. Status code: {upload_status}")

            return "upload successful"

        else:
            # Log failure if no presigned URLs could be obtained
            print(f"Failed to get presigned URLs.")
            return None

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
                print("Token obtained successfully.")
                await self.check_and_upload_files(session, file_paths=file_names)
            else:
                print("Failed to obtain token.")
