import argparse
import Crawler.model_dev.fpredict as fp
import pandas as pd

TIMECODES = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']


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
    parser.add_argument('-m', '--model', dest='model',
                        type=str, help='Input exisiting model folder',
                        metavar='')
    arg = parser.parse_args()
    return arg.code, arg.file, arg.time, arg.model


def main():
    code, filename, time, model = parse_arguments()
    # Use CSV to retrieve info
    # df = pd.read_csv('LENDLEASE info max.csv')
    x = fp.Crawl(code, output_graph=True)
    x.model_dev()
    # x = fp.Crawl(code, model_name=model)
    # print(x.model_predict())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt.")
