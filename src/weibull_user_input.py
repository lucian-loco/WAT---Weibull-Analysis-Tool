#!/usr/bin/python3
"""
weibull_user_input.py
=======================

Interactive command-line input helpers for the Weibull reliability analysis toolset.

This module collects and validates user input for the parameters needed by the automated and manual Weibull analysis
workflows in `weibull_analysis.py` and the forecast CLI in `weibull_forecast.py`:
failure/distinct-failure thresholds, the model-selection method (AICc/BIC/CV), the confidence interval level,
and forecast time horizons.

Each `ask_*` function implements a blocking input-validation loop:
it repeatedly prompts the user via `input()`, displays an inline error message and re-prompts on invalid entries,
accepts a blank entry as "use the default value", and only returns once a valid value has been provided.
These functions are intended for direct terminal use (e.g. in each analysis script's `if __name__ == "__main__":` block)
 rather than for programmatic/headless use.

Note: the validation logic here overlaps with (and is a CLI-interactive counterpart to) the `validate_*` functions in
`utils.py`, which return (value, error_message) tuples for use in non-blocking contexts (e.g. a web form) rather than
looping on `input()`.

Author: Lucian Groha
"""



def ask_threshold(name: str, default: int):
    """
    Interactively prompt the user for an integer threshold value (e.g. minimum failure count or
    minimum distinct-failure count), looping until valid input is given.

    Parameters
    ----------
    name : str
        Human-readable label for the threshold being requested, inserted into the prompt text
        (e.g. "Failure threshold", "Distinct threshold").
    default : int
        Value returned if the user submits a blank entry.

    Returns
    -------
    int
        The user-provided integer (must be > 1) or `default` if the input was left blank.

    Notes
    -----
    Blocks on `input()` in a loop, printing an inline error message and re-prompting on non-integer input or values <= 1.
    There is no exit condition other than providing valid input.
    """
    while True:
        user_input = input(f"Enter {name} and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        try:
            value = int(user_input)
            if value > 1:
                return value
            else:
                print("  → Please enter a positive number greater than 1.")
        except ValueError:
            print("  → Invalid input, please enter an integer..")


def ask_sort_by(default: str = 'BIC'):
    """
    Interactively prompt the user to choose the distribution-selection method used by `compare_best_distribution`,
    looping until a valid option is given.

    Valid options
    -------------
    - 'AICc' : pure information-criterion selection (AICc/BIC rule).
    - 'BIC'  : pure information-criterion selection (AICc/BIC rule).
    - 'CV'   : cross-validation-driven selection, with an IC fallback if CV is infeasible.

    Parameters
    ----------
    default : str, optional
        Value returned if the user submits a blank entry (default: 'BIC'). The prompt text notes 'CV' as the generally
        recommended choice, independent of this fallback default.

    Returns
    -------
    str
        One of 'AICc', 'BIC', or 'CV' — either the user's valid input or `default` if left blank.

    Notes
    -----
    Blocks on `input()` in a loop, printing an inline error message and re-prompting on any input
    not in `['AICc', 'BIC', 'CV']`.
    """
    valid_options = ['AICc', 'BIC', 'CV']
    while True:
        user_input = input(
            f"Enter selection method {valid_options} and press enter (Recommended: CV, but default value as fallback is {default}): "
        ).strip()
        if user_input == "":
            return default
        if user_input in valid_options:
            return user_input
        else:
            print(f" → Invalid input, please enter one of {valid_options}.")


def ask_ci(default: float = 0.95):
    """
    Interactively prompt the user for a confidence interval level, looping until valid input is given.

    Parameters
    ----------
    default : float, optional
        Value returned if the user submits a blank entry (default: 0.95).

    Returns
    -------
    float
        The user-provided confidence level in the range [0, 1), or `default` if the input was left blank.
        A value of 0 is treated as a valid, meaningful choice representing "no confidence interval".

    Notes
    -----
    Blocks on `input()` in a loop, printing an inline error message and re-prompting on non-numeric input or
    values outside [0, 1) (1 itself is rejected as an invalid confidence level).
    """
    while True:
        user_input = input(f"Enter confidence interval [0-1) and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        try:
            value = float(user_input)
            if 0 <= value < 1:
                return value
            else:
                print("  → Please enter a value strictly between 0 and 1 or 0 for no confidence interval.")
        except ValueError:
            print("  → Invalid input, please enter a number (e.g. 0.95).")


def ask_deltas(default: list = None):
    """
    Interactively prompt the user for a comma-separated list of future forecast time horizons (in days),
    looping until valid input is given.

    Parameters
    ----------
    default : list[float], optional
        Values returned if the user submits a blank entry. If not provided, defaults to [90.0, 180.0, 365.0].
        Displayed to the user in the prompt as a comma-separated string (integers shown without a decimal point,
        e.g. "90,180,365").

    Returns
    -------
    list[float]
        The user-provided list of positive forecast horizons in days, or `default` if the input was left blank.

    Notes
    -----
    Blocks on `input()` in a loop, printing an inline error message and re-prompting if the input cannot be parsed
    as comma-separated numbers, is empty, or contains any value <= 0. When the default is used,
    an explicit confirmation message is printed before returning.
    """
    if default is None:
        default = [90.0, 180.0, 365.0]
    default_str = ','.join(str(int(d) if d == int(d) else d) for d in default)
    while True:
        user_input = input(f"Enter comma-separated future time deltas in days and press enter (Default: {default_str}): ").strip()
        if user_input == "":
            print(f"Default ({default_str}) will be used.")
            return default
        try:
            values = [float(x.strip()) for x in user_input.split(',')]
            if not values:
                print(" → Please enter at least one value.")
                continue
            if any(v <= 0 for v in values):
                print(" → All delta values must be positive numbers greater than 0.")
                continue
            return values
        except ValueError:
            print(" → Invalid input, please enter comma-separated numbers (e.g. 90,180,365).")