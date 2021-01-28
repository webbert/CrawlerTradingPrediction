import argparse
import yfinance as yf
import crawler.fpredict as fp
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Trading prediction and bot tool")
    parser.add_argument('-c', '--code', type=str,
                        dest='code', help="Code based on YAHOO Finance.")
    arg = parser.parse_args()
    return arg.code


def main(code):
    # Retrieve pandas dataframe information of stocks/REITs
    yf_object = yf.Ticker(code)
    max_info = yf_object.history(period="1Y")

    # Use CSV to retrieve info
    # df = pd.read_csv('LENDLEASE info max.csv')
    x = fp.predict_obj(max_info)
    # x.graph()
    x.predict_test_3(60, code)
    # x.write_to_csv("LENDLEASE info max")


if __name__ == '__main__':
    try:
        code = parse_arguments()
        main(code)
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt.")
