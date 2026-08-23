"""Expected application errors for the command-line boundary."""


class ApplicationError(Exception):
    """Base class for expected failures that can be shown to a user."""


class ConfigurationError(ApplicationError):
    """Raised when application configuration is missing or invalid."""


class DocumentError(ApplicationError):
    """Raised when a supported document cannot be read or processed."""


class PersistenceError(ApplicationError):
    """Raised when local persistence cannot safely complete."""


class EmbeddingError(ApplicationError):
    """Raised when local embedding generation cannot safely complete."""


class RetrievalError(ApplicationError):
    """Raised when persisted embeddings cannot be safely compared."""


class PromptError(ApplicationError):
    """Raised when grounded prompt preparation receives invalid input."""
