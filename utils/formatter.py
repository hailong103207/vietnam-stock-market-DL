import datetime
import time

class Formatter:
    @staticmethod
    def to_timestamp(date_str: str, date_format: str = "%d-%m-%Y") -> int:
        dt = datetime.datetime.strptime(date_str, date_format)
        return int(time.mktime(dt.timetuple()))

if __name__ == "__main__":
    date_str = "2023-10-20"
    timestamp = Formatter.to_timestamp(date_str)
    print(f"Timestamp for {date_str} is {timestamp}")