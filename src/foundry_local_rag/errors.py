"""Expected application errors for the command-line boundary."""


class ApplicationError(Exception):
    """Base class for expected failures that can be shown to a user."""


class ConfigurationError(ApplicationError):
    """Raised when application configuration is missing or invalid."""
