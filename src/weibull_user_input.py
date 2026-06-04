#!/usr/bin/python3
# Functions to deal with user inputs


def ask_threshold(name: str, default: int):
    """
    Asks the user to choose the threshold for the failure count and the distinct failures.
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
    Asks the user to choose the sort by / selection method.
    Valid options:
      - 'AICc'  → pure information-criterion selection (AICc/BIC rule)
      - 'BIC'   → pure information-criterion selection (AICc/BIC rule)
      - 'CV'    → cross-validation-driven selection (with IC fallback)
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
    Asks the user to choose the ci level.
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
    Asks the user to enter comma-separated future time deltas in days.
    Validates that all values are positive numbers.
    Returns a list of floats.
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