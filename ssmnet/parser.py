from argparse import ArgumentParser

def parse_args():
    
    parser = ArgumentParser(description='Compute SSM, novelty and boundaries using SSM-Net')
    parser.add_argument("audio_file", 
                        help='audio file to process')
    parser.add_argument("-o", "--output_csv_file", default="output.csv",
                        help='output csv file that contains the boundary positions [in sec]')
    parser.add_argument("-p", "--output_pdf_file", default="output.pdf",
                        help='output pdf file with SSM, novelty-curve and detected boundaries')
    parser.add_argument("-c", "--config_file", default="config_example.yaml",
                        help='fullpath to a yaml configuration file')
    
    return parser.parse_args()
