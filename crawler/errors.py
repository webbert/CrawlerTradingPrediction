"""
File for custom errors.
"""

TIMECODES = ['1D', '5D', '1M', '6M', 'YTD', '1Y', '5Y', 'MAX']


class YahooFinanceCodeDoesNotExist(Exception):
    def __init__(self, timespan, timecodes):
        self.timespan = timespan
        self.timecodes = timecodes
        self.message = (f"{self.timespan} not one of the choices\nPlease Choose\
 the following options only: {TIMECODES}")
        super().__init__(self.message)


class DaysInputError(Exception):
    def __init__(self, days):
        self.days = days
        self.message = (f"{self.days} cannot be used in the function.")
        super().__init__(self.message)
