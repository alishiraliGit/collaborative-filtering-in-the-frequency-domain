from datetime import datetime
import pickle
from matplotlib import pyplot as plt
import os


class Logger:
    def __init__(self, settings, save_path, do_plot=False):
        self.settings = settings
        self.save_path = save_path
        self.do_plot = do_plot

        self.rmse_tr = []
        self.rmse_va = []
        self.rmse_te = []

    def log(self, rmse_tr, rmse_va, rmse_te):
        self.rmse_tr += [rmse_tr]
        self.rmse_va += [rmse_va]
        self.rmse_te += [rmse_te]

        print('iteration: %d, rmse train: %.3f, rmse val: %.3f rmse test: %.3f' %
              (len(self.rmse_tr), self.rmse_tr[-1], self.rmse_va[-1], self.rmse_te[-1]))

        if self.do_plot:
            plt.plot(len(self.rmse_tr), self.rmse_tr[-1], 'ro')
            plt.plot(len(self.rmse_va), self.rmse_va[-1], 'ko')
            plt.plot(len(self.rmse_te), self.rmse_te[-1], 'bo')
            plt.legend(['train', 'validation', 'test'])
            plt.ylabel('rmse')
            plt.xlabel('#iter')
            plt.pause(0.05)

    def save(self, ext=None):
        stringified = 'result' + Logger.stringify(self.settings)
        if len(stringified) > 200:
            stringified = stringified[:200]

        file_name = stringified + '-' + datetime.now().strftime('%Y-%m-%d %H-%M-%S')

        with open(os.path.join(self.save_path, file_name + '.res'), 'wb') as f:
            pickle.dump({'rmse_tr': self.rmse_tr,
                         'rmse_va': self.rmse_va,
                         'rmse_te': self.rmse_te,
                         'settings': self.settings}, f)

        if ext is not None:
            with open(os.path.join(self.save_path, file_name + '.extres'), 'wb') as f:
                pickle.dump(ext, f)

    @staticmethod
    def load(load_path, file_name, load_ext=False):
        file_path = os.path.join(load_path, file_name)
        if load_ext:
            file_path += '.extres'
        else:
            file_path += '.res'

        with open(file_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def stringify(dic):
        stringified = ''
        for key, val in dic.items():
            stringified += '-'
            stringified += key
            if isinstance(val, int):
                stringified += str(val)
            elif isinstance(val, float):
                stringified += '%.0e' % val
            elif isinstance(val, str):
                stringified += val
            else:
                stringified += Logger.stringify(val)

        return stringified

    @staticmethod
    def dicts_to_dic(dicts):
        items_list = []
        for dic in dicts:
            items_list += list(dic.items())

        return dict(items_list)
