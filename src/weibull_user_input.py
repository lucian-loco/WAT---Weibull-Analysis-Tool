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
    Asks the user to choose the sort by.
    """
    valid_options = ['AICc', 'BIC']
    while True:
        user_input = input(f"Enter sort method {valid_options} and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        if user_input in valid_options:
            return user_input
        else:
            print(f"  → Invalid input, please enter one of {valid_options}.")


def ask_ci(default: float = 0.95):
    """
    Asks the user to choose the ci level.
    """
    while True:
        user_input = input(f"Enter confidence interval (0-1) and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        try:
            value = float(user_input)
            if 0 < value < 1:
                return value
            else:
                print("  → Please enter a value strictly between 0 and 1.")
        except ValueError:
            print("  → Invalid input, please enter a number (e.g. 0.95).")