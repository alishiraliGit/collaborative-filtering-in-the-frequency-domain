from datetime import datetime
import pickle
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, settings, save_path, do_plot=False):
        self.settings = settings
        self.save_path = save_path
        self.do_plot = do_plot

        self.rmse_tr = []
        self.rmse_te = []

    def log(self, rmse_tr, rmse_te):
        self.rmse_tr += [rmse_tr]
        self.rmse_te += [rmse_te]

        print('iteration: %d, rmse train: %.3f, rmse test: %.3f' %
              (len(self.rmse_tr), self.rmse_tr[-1], self.rmse_te[-1]))

        if self.do_plot:
            plt.plot(len(self.rmse_tr), self.rmse_tr[-1], 'ro')
            plt.plot(len(self.rmse_te), self.rmse_te[-1], 'bo')
            plt.pause(0.05)

    def save(self):
        stringified = 'result'
        for key, val in self.settings.items():
            stringified += '-'
            stringified += key
            if isinstance(val, int):
                stringified += str(val)
            elif isinstance(val, float):
                stringified += '%.3f' % val
            elif isinstance(val, str):
                stringified += val
            else:
                raise Exception('Unknown value')

        file_name = stringified + '-' + datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        with open(self.save_path + '/' + file_name + '.res', 'wb') as f:
            pickle.dump({'rmse_tr': self.rmse_tr,
                         'rmse_te': self.rmse_te,
                         'settings': self.settings}, f)

    @staticmethod
    def load(load_path, file_name):
        with open(load_path + '/' + file_name + '.res', 'rb') as f:
            return pickle.load(f)
