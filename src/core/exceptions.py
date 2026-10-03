class ApplicationError(Exception):
    """
    Base exception for expected application-level errors.
    """

    status_code = 500
    default_detail = "Application error"

    def __init__(
        self,
        detail: str | None = None,
    ):
        self.detail = detail or self.default_detail

        super().__init__(self.detail)


class BadRequestError(ApplicationError):
    """
    Raised when the request or input violates
    an application-level rule.
    """

    status_code = 400
    default_detail = "Bad request"


class ServiceUnavailableError(ApplicationError):
    """
    Raised when an external dependency is temporarily
    unavailable after retry attempts are exhausted.
    """

    status_code = 503
    default_detail = "Service temporarily unavailable"
