def get_hms(seconds):
    """
    Convert seconds to hours, minutes, and seconds.

    Parameters:
    seconds (int): Number of seconds to be converted.

    Returns:
    tuple: A tuple containing hours, minutes, and seconds.
    """

    hours = seconds // 3600
    remaining_seconds = seconds % 3600
    minutes = remaining_seconds // 60
    remaining_seconds = remaining_seconds % 60

    return hours, minutes, remaining_seconds
