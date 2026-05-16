class WorkerError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra


def error_payload(exc):
    payload = {"code": exc.code, "message": exc.message}
    payload.update(exc.extra)
    return payload
