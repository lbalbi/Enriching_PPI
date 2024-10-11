import torch

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) > 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result[:5])

    # faz print para cada hits@K
    def print_statistics(self, run=None):

        if run is not None:
            # todos os resultados para cada run (cada tensor possui os eval scores de todas as epochs)
            result = 100 * torch.tensor(self.results[run]) # aspeto: tensor([[3.5 ,0.1, 0.1,68.0,87.8,84.5], [ 1.5, 0.1, 0.1, 68.9, 93.6, 83.5]])
            argmax = result[:, 1].argmax().item() # result[:, 1] vai à 2a coluna, dos scores de validation, buscar o idx do mais elevado
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')  # teste final tem em conta o melhor validation score
            # acrescentado
            if len(result[argmax,]) > 3:
                print(f'  Highest Accuracy: {result[:, 3].max():.2f}') # para ter ideia do Roc-Auc mais elevado obtido
                print(f'  Final Accuracy: {result[argmax, 3]:.2f}') # rocauc final a ter em conta melhor validation score
                print(f'  Highest F1-score: {result[:, 4].max():.2f}')
                print(f'  Final F1-score: {result[argmax, 4]:.2f}')

        else:
            result = 100 * torch.tensor(self.results)
            # melhores resultados de todas as runs
            best_results = []
            for r in result:
            # cada r é uma run, para cada run obtém-se o best score
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                # acrescentado
                if len(r[0]) > 3:
                    acc = r[r[:, 1].argmax(), 3].item()
                    f1score = r[r[:, 1].argmax(), 4].item()
                    # best score de cada run é guardado
                    best_results.append((train1, valid, train2, test, acc, f1score))
                else:
                    best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)
            # média dos best scores de todas as runs
            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            # acrescentado
            if len(best_result[0, :]) > 4:
                r = best_result[:, 4]
                print(f'   Final Accuracy: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 5]
                print(f'   Final F1-score: {r.mean():.2f} ± {r.std():.2f}')

