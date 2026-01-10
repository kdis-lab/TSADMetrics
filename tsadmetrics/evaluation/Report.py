import pandas as pd

class Report:
    def __init__(self):
        pass
    def generate_report(self, results, output_file):
        """
        Generate a report from the evaluation results.

        Parameters:
            results (dict):
                Dictionary containing evaluation results.
            output_file (str):
                Path to the output file where the report will be saved.
        """
        
        if type(results) is dict:
            df = pd.DataFrame.from_dict(results, orient='index')

            df.index.name = 'dataset'
            df.reset_index(inplace=True)

            df.to_csv(output_file, index=False, sep=';')
        else:
            results.to_csv(output_file, sep=';')