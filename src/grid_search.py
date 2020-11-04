
import pandas, numpy, uuid, os
from multiprocessing import Process, Pool
from matplotlib import pyplot
from src.torch import adversarial


def run_parallel(grid):
    """ Not working because of some weird Catalina error. """

    chunks = []
    for i in range(0, grid.shape[0], 3):
        chunk = []
        for j in range(i, i+3):

            if j >= grid.shape[0]:
                pass
            else:
                parameters = dict(grid.iloc[i,:])
                parameters['id'] = str(uuid.uuid4())[:8]
                chunk.append(parameters)
        chunks.append(chunk)

    for chunk in chunks:

        p1 = Process(target=adversarial.main, args=(chunk[0],))
        p2 = Process(target=adversarial.main, args=(chunk[1],))
        p3 = Process(target=adversarial.main, args=(chunk[2],))

        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()


if __name__ == "__main__":

    grid = pandas.read_csv('/Users/andreidm/ETH/projects/normalization/data/test_grid.csv', sep=';', index_col=0).dropna()

    for i in range(grid.shape[0]):
        parameters = dict(grid.iloc[i, :])
        parameters['id'] = str(uuid.uuid4())[:8]
        adversarial.main(parameters)

    path = '/Users/andreidm/ETH/projects/normalization/res/callbacks/'
    for file in os.listdir(path):
        if file.startswith('history'):
            # check results
            history = pandas.read_csv(path + file)
            print(file)
            find_best_epoch(history)
            print()


