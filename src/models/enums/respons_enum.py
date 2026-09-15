from enum import Enum


class ResponseSignal(Enum):
    FILE_VALIDATED_SUCCESS = "File validated successfully"
    FILE_SIZE_EXCEEDED = "File size exceeds the allowed limit"
    FILE_TYPE_NOT_SUPPORTED = "File type is not supported"
    FILE_UPLOAD_FAILED = "File upload failed"
    FILE_UPLOAD_SUCCESS = "File uploaded successfully"
    FILE_ID_ERROR = "no_file_found_with_this_id"
    NO_FILES_ERROR = "not_found_files"
    PROCESSING_FAILED = "Processing failed"
    PROCESSING_SUCCESS = "processing_success"
