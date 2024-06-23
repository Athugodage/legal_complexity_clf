import configparser
import argparse
from main import BlackBox

config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default=config['OLD_PIPE']['input_path'])
    parser.add_argument('--bert_path', type=str, default=config['OLD_PIPE']['bert_path'])
    parser.add_argument('--cluster_path', type=str, default=config['OLD_PIPE']['cluster_model'])
    parser.add_argument('--graph_path', type=str, default=config['OLD_PIPE']['graph_model'])
    parser.add_argument('--saved_data_path', type=str, default="saved_data")
    parser.add_argument('--cp_path', type=str, default=config['OLD_PIPE']['model_state'])
    parser.add_argument('--kad_path', type=str)
    parser.add_argument('--save_report', type=str, default=config['OLD_PIPE']['full_report'])
    parser.add_argument('--save_prediction_path', type=str, default=config['OLD_PIPE']['prediction_file'])

    args = parser.parse_args()
    KAD_CAT_INFO_PATH = args.kad_path

    pipe = BlackBox(bert_path=args.bert_path,
                    cluster_path=args.cluster_path,
                    graph_path=args.graph_path,
                    saved_data_path=args.saved_data_path,
                    cp_path=args.cp_path,
                    kad_path=args.kad_path,
                    )

    result = pipe.implement(input_path=args.input_path,
                            save_report=args.save_report,
                            save_prediction_path=args.save_prediction_path)
    print(result)