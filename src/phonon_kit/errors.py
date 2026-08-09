class PhononKitError(RuntimeError):
    """Base exception shown as a concise CLI error."""


class ConfigError(PhononKitError):
    """Invalid or inconsistent user configuration."""


class RunStateError(PhononKitError):
    """A run cannot safely be created or resumed."""


class ExternalProgramError(PhononKitError):
    """DeepMD, VASP, or DPDispatcher failed."""


class IncompleteResultsError(PhononKitError):
    """Required force-calculator results are not complete yet."""

