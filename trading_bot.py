import argparse
import Crawler.fpredict as fp
import pandas as pd

TIMECODES = ['1D', '5D', '1M', '6M', 'YTD', '1Y', '5Y', 'MAX']


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Trading prediction and bot tool")
    m_group = parser.add_mutually_exclusive_group()
    m_group.add_argument('-c', '--code', type=str,
                         dest='code', help="Code based on YAHOO Finance.",
                         metavar="", default=None)
    m_group.add_argument('-f', '--file', type=str,
                         dest='file', help='If CSV file present to be parsed.',
                         default=None)
    parser.add_argument('-t', '--time',
                        dest="time", choices=TIMECODES,
                        help=f"The time spans.\nChoices are as followed: \
                            {TIMECODES}", metavar="")
    arg = parser.parse_args()
    return arg.code, arg.file, arg.time


def main():
    code, filename, time = parse_arguments()
    # Use CSV to retrieve info
    # df = pd.read_csv('LENDLEASE info max.csv')
    x = fp.Crawl(code)
    x.predict_()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt.")
