"""
Custom errors.
"""

TIMECODES = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


class YahooFinanceCodeDoesNotExist(Exception):
    def __init__(self, timespan, timecodes):
        self.timespan = timespan
        self.timecodes = timecodes
        self.message = (f"{self.timespan} not one of the choices\nPlease "
                        f"Choose the following options only: {TIMECODES}")
        super().__init__(self.message)


class DaysInputError(Exception):
    def __init__(self, days, len_data):
        self.days = days
        self.message = (
            f"{self.days} is too many days for the data given of length"
            f" {len_data}.")
        super().__init__(self.message)
