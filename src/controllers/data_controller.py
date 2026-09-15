import os
import re

from fastapi import UploadFile

from controllers.base_controller import BaseController
from models import ResponseSignal

from .project_controller import ProjectController


class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.scale = 1048576

    def validate_uploaded_file(self, file: UploadFile):
        print("filename:", file.filename)
        print("content_type:", file.content_type)

        # Check that a file was provided
        if file is None:
            return False, ResponseSignal.FILE_NOT_FOUND.value

        # Validate file type
        if file.content_type not in self.app_setting.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value

        # Validate file size
        max_size = self.app_setting.FILE_MAX_SIZE * self.scale

        if file.size is not None and file.size > max_size:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value

        return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value

    def cleaned_file_name(self, orig_file_name: str):
        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r"[^\w.]", "", orig_file_name.strip())

        # replace spaces with underscore
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name

    def generate_unique_filepath(self, orig_file_name: str, project_id: str):
        project_path = ProjectController().get_project_path(project_id=project_id)

        clean_file_name = self.cleaned_file_name(orig_file_name=orig_file_name)

        while True:
            random_key = self.generate_random_string()
            file_name = f"{random_key}_{clean_file_name}"
            file_path = os.path.join(project_path, file_name)

            if not os.path.exists(file_path):
                return file_path, file_name
