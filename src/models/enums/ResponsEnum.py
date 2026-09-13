from enum import Enum


class ResponseSignal(Enum):
    FILE_VALIDATED_SUCCESS = "File validated successfully"
    FILE_SIZE_EXCEEDED = "File size exceeds the allowed limit"
    FILE_TYPE_NOT_SUPPORTED = "File type is not supported"
    FILE_UPLOAD_FAILED = "File upload failed"
    FILE_UPLOAD_SUCCESS = "File uploaded successfully"
