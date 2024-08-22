import os
import datetime
import logging
import asyncio
import aiohttp
import requests
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from .exporters.file_span_exporter import FileSpanExporter
from .exporters.raga_exporter import RagaExporter
from .instrumentators import (
    LangchainInstrumentor,
    OpenAIInstrumentor,
    LlamaIndexInstrumentor,
)
from .utils import get_unique_key

logger = logging.getLogger(__name__)


class Tracer:

    def __init__(
        self,
        project_name,
        tracer_type=None,
        pipeline=None,
        metadata=None,
        description=None,
        upload_timeout=30,  # Default timeout of 30 seconds
    ):
        """
        Initializes a Tracer object.

        Args:
            project_name (str): The name of the project.
            tracer_type (str, optional): The type of tracer. Defaults to None.
            pipeline (dict, optional): The pipeline configuration. Defaults to None.
            metadata (dict, optional): The metadata. Defaults to None.
            description (str, optional): The description. Defaults to None.
            upload_timeout (int, optional): The upload timeout in seconds. Defaults to 30.

        Returns:
            None
        """
        self.project_name = project_name
        self.tracer_type = tracer_type
        self.metadata = self._improve_metadata(metadata, tracer_type)
        self.pipeline = pipeline
        self.description = description
        self.upload_timeout = upload_timeout

        self.raga_client = RagaExporter(project_name=self.project_name)

        self._tracer_provider = self._setup_provider()
        self._instrumentor = self._setup_instrumentor(tracer_type)
        self.filespanx = None
        self.is_instrumented = False
        self._upload_task = None

    def _improve_metadata(self, metadata, tracer_type):
        if metadata is None:
            metadata = {}
        metadata.setdefault("log_source", f"{tracer_type}_tracer")
        metadata.setdefault("recorded_on", str(datetime.datetime.now()))
        return metadata

    def _add_unique_key(self, data, key_name):
        data[key_name] = get_unique_key(data)
        return data

    def _setup_provider(self):
        self.filespanx = FileSpanExporter(
            project_name=self.project_name,
            metadata=self.metadata,
            pipeline=self.pipeline,
            raga_client=self.raga_client,
        )
        self.tracer_provider = trace_sdk.TracerProvider()
        self.tracer_provider.add_span_processor(SimpleSpanProcessor(self.filespanx))
        return self.tracer_provider

    def _setup_instrumentor(self, tracer_type):
        instrumentors = {
            "langchain": LangchainInstrumentor,
            "openai": OpenAIInstrumentor,
            "llama_index": LlamaIndexInstrumentor,
        }
        if tracer_type not in instrumentors:
            raise ValueError(f"Invalid tracer type: {tracer_type}")
        return instrumentors[tracer_type]().get()

    @contextmanager
    def trace(self):
        """
        Synchronous context manager for tracing.
        Usage:
            with tracer.trace():
                # Your code here
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def start(self):
        """Start the tracer."""
        if not self.is_instrumented:
            self._setup_provider()
            self._instrumentor().instrument(tracer_provider=self._tracer_provider)
            self.is_instrumented = True
        print(f"Tracer started for project: {self.project_name}")
        return self

    def stop(self):
        """Stop the tracer and initiate trace upload."""
        if not self.is_instrumented:
            logger.warning("Tracer was not started. No traces to upload.")
            return "No traces to upload"

        print("Stopping tracer and initiating trace upload...")
        self._cleanup()
        self._upload_task = self._run_async(self._upload_traces())
        return "Trace upload initiated. Use get_upload_status() to check the status."

    def get_upload_status(self):
        """Check the status of the trace upload."""
        if self._upload_task is None:
            return "No upload task in progress."
        if self._upload_task.done():
            try:
                result = self._upload_task.result()
                return f"Upload completed: {result}"
            except Exception as e:
                return f"Upload failed: {str(e)}"
        return "Upload in progress..."

    def _run_async(self, coroutine):
        """Run an asynchronous coroutine in a separate thread."""
        loop = asyncio.new_event_loop()
        with ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: loop.run_until_complete(coroutine))
        return future

    async def _upload_traces(self):
        """
        Asynchronously uploads traces to the RagaAICatalyst server.

        This function uploads the traces generated by the RagaAICatalyst client to the RagaAICatalyst server. It uses the `aiohttp` library to make an asynchronous HTTP request to the server. The function first checks if the `RAGAAI_CATALYST_TOKEN` environment variable is set. If not, it raises a `ValueError` with the message "RAGAAI_CATALYST_TOKEN not found. Cannot upload traces.".

        The function then uses the `asyncio.wait_for` function to wait for the `check_and_upload_files` method of the `raga_client` object to complete. The `check_and_upload_files` method is called with the `session` object and a list of file paths to be uploaded. The `timeout` parameter is set to the value of the `upload_timeout` attribute of the `Tracer` object.

        If the upload is successful, the function returns the string "Files uploaded successfully" if the `upload_stat` variable is truthy, otherwise it returns the string "No files to upload".

        If the upload times out, the function returns a string with the message "Upload timed out after {self.upload_timeout} seconds".

        If any other exception occurs during the upload, the function returns a string with the message "Upload failed: {str(e)}", where `{str(e)}` is the string representation of the exception.

        Parameters:
            None

        Returns:
            A string indicating the status of the upload.
        """
        async with aiohttp.ClientSession() as session:
            if not os.getenv("RAGAAI_CATALYST_TOKEN"):
                raise ValueError(
                    "RAGAAI_CATALYST_TOKEN not found. Cannot upload traces."
                )

            try:
                upload_stat = await asyncio.wait_for(
                    self.raga_client.check_and_upload_files(
                        session=session,
                        file_paths=[self.filespanx.sync_file] if self.filespanx else [],
                    ),
                    timeout=self.upload_timeout,
                )
                return (
                    "Files uploaded successfully"
                    if upload_stat
                    else "No files to upload"
                )
            except asyncio.TimeoutError:
                return f"Upload timed out after {self.upload_timeout} seconds"
            except Exception as e:
                return f"Upload failed: {str(e)}"

    def _cleanup(self):
        """
        Cleans up the tracer by uninstrumenting the instrumentor, shutting down the tracer provider,
        and resetting the instrumentation flag. This function is called when the tracer is no longer
        needed.

        Parameters:
            self (Tracer): The Tracer instance.

        Returns:
            None
        """
        if self.is_instrumented:
            try:
                self._instrumentor().uninstrument()
                self._tracer_provider.shutdown()
                print("Tracer provider shut down successfully")
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {str(e)}")

        # Reset instrumentation flag
        self.is_instrumented = False
        self.filespanx = None