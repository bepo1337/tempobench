import datetime


def date_string_to_datetime(date_str) -> datetime.datetime:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d")