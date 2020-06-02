import os

def results_gathering():
    dir = 'data/'
    with open('data/all_data.csv', 'w') as final:
        header = 'Model,dataset,Size,MSE,RMSE,Concordance Index,Spearman,Pearson'
        print(header)
        final.write(header+'\n')
        for subdir, dirs, files in os.walk(dir):
            for filename in files:
                filepath = subdir + os.sep + filename
                if filepath.endswith("best_metrics.csv"):
                    with open(filepath) as file:
                        file.readline()
                        metrics = file.readline()
                        print(metrics)
                        final.write(metrics+'\n')

if __name__=="__main__":
    results_gathering()