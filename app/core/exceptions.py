from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class UnsupportedFileTypeError(AppError):
    status_code = 415
    detail = "Only .pdf and .txt files are supported"


class FileTooLargeError(AppError):
    status_code = 413
    detail = "File is too large"


class EmptyDocumentError(AppError):
    status_code = 422
    detail = "No extractable text found in the document"


class LLMUnavailableError(AppError):
    status_code = 503
    detail = "The language model is unavailable"
