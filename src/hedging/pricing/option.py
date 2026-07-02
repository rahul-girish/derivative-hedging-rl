from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionType(str, Enum):
    """Supported European option types."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class OptionContract:
    """
    Represents a European vanilla option.

    Attributes
    ----------
    strike:
        Strike price.

    maturity:
        Time to maturity in years.

    option_type:
        Either CALL or PUT.
    """

    strike: float
    maturity: float
    option_type: OptionType

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("Strike price must be positive.")

        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")