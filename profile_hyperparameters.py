import random
import pandas as pd
import numpy as np
from kTourSelection import SimpleGeneticAlgorithm

if __name__ == '__main__':

    #macro_alpha_list = [0.3,0.4,0.5,0.6,0.7]
    #micro_alpha_list = [0.01,0.05,0.1,0.15,0.2]
    macro_alpha_list = [0.3,0.4]
    micro_alpha_list = [0.01,0.05]

    results = []

    for run_num in range(0,4):
        for macro_alpha in macro_alpha_list:
            print('Working on macro: '+str(macro_alpha))

            df = pd.read_csv('tour29.csv', header=None)
            DistanceMatrix = []
            for i in range(df.shape[0]):
                row = []
                for j in range(df.shape[1]):
                    row.append(df.iloc[i, j])
                DistanceMatrix.append(row)

            start_point = 0
            EA = SimpleGeneticAlgorithm(DistanceMatrix, start_point, macro_alpha=macro_alpha)
            best_path, best_result = SimpleGeneticAlgorithm.Iteration(EA)
            results.append({'macro':macro_alpha, 'run_num':run_num,'result': best_result})

    df_res = pd.DataFrame(results)
    df_res.to_csv('Macro_comparative.csv', index=False)