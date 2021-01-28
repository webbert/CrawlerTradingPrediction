import argparse
import crawler.fpredict as fp
import pandas as pd

TIMECODES = ['1D', '5D', '1M', '6M', 'YTD', '1Y', '5Y', 'MAX']


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Trading prediction and bot tool")

    parser.add_argument('-c', '--code', type=str,
                        dest='code', help="Code based on YAHOO Finance.",
                        required=True, metavar="")
    parser.add_argument('-t', '--time',
                        dest="time", choices=TIMECODES,
                        help=f"The time spans.\nChoices are as followed: \
                            {TIMECODES}", metavar="")
    arg = parser.parse_args()
    return arg.code, arg.time


def main():
    code, time = parse_arguments()
    # Use CSV to retrieve info
    # df = pd.read_csv('LENDLEASE info max.csv')
    x = fp.Crawl(code, time)
    x.predict_(60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt.")
