import argparse
import math
import os.path as osp
import pickle
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True, type=str, help='Path to input csv file to translate.')
    parser.add_argument('-o', '--output_file', required=True, type=str, help='Path to output edgelist_file.')
    # parser.add_argument('--tokenization', required=True, type=str, choices=['token', 'flatten', 'all'],
    #                     help='Tokenization strategy to use.')
    parser.add_argument('--info_file', required=False, type=str, default=None,
                        help='Path to info file with df boundaries.')

    parser.add_argument('--export', required=False, action='store_true',
                        help='Flag for exporting the edgelist in networkx format.')

    return parser.parse_args()


class EdgeList:
    @staticmethod
    def f_no_smoothing():
        return 1.0

    @staticmethod
    def smooth_exp(x, eps=0.01, target=10, k=0.5):
        t = (eps / (1 - k)) ** (1 / (1 - target))
        y = (1 - k) * t ** (-x + 1) + k
        return y

    @staticmethod
    def inverse_smooth(x, s):
        y = 1 / 2 * (-(1 + s) ** (1 - x) + 2)
        return y

    @staticmethod
    def inverse_freq(freq):
        return 1 / freq

    @staticmethod
    def log_freq(freq, base=10):
        return 1 / (math.log(freq, base) + 1)

    def smooth_freq(self, freq, eps=0.01):
        if self.smoothing_method == 'smooth':
            return self.smooth_exp(freq, eps, self.smoothing_target)
        if self.smoothing_method == 'inverse_smooth':
            return self.inverse_smooth(freq, self.smoothing_k)
        elif self.smoothing_method == 'log':
            return self.log_freq(freq, self.smoothing_target)
        elif self.smoothing_method == 'inverse':
            return self.inverse_freq(freq)
        elif self.smoothing_method == 'no':
            return self.f_no_smoothing()

    @staticmethod
    def convert_cell_value(original_value):
        '''
        Convert cell values to strings. Round float values.
        :param original_value: The value to convert to str.
        :return: The converted value.
        '''

        # If the cell is the empty string, or np.nan, return None.
        if original_value == '':
            return None
        if original_value != original_value:
            return None
        try:
            float_c = float(original_value)
            if math.isnan(float_c):
                return None
            cell_value = str(int(float_c))
        except ValueError:
            cell_value = str(original_value)
        except OverflowError:
            cell_value = str(original_value)
        return cell_value

    def _parse_smoothing_method(self, smoothing_method):
        '''
        Convert the smoothing method supplied by the user into parameters that can be used by the edgelist.
        This function is also performing error checking on the input parameters.

        :param smoothing_method: One of "smooth", "inverse_smooth", "log", "piecewise", "no".
        '''
        if smoothing_method.startswith('smooth'):
            smooth_split = smoothing_method.split(',')
            if len(smooth_split) == 3:
                self.smoothing_method, self.smoothing_k, self.smoothing_target = smooth_split
                self.smoothing_k = float(self.smoothing_k)
                self.smoothing_target = float(self.smoothing_target)
                if not 0 <= self.smoothing_k <= 1:
                    raise ValueError('Smoothing k must be in range [0,1], current k = {}'.format(self.smoothing_k))
                if self.smoothing_target < 1:
                    raise ValueError('Smoothing target must be > 1, current target = {}'.format(self.smoothing_target))
            elif len(smooth_split) == 1:
                self.smoothing_method = 'smooth'
                self.smoothing_k = 0.2
                self.smoothing_target = 200
            else:
                raise ValueError('Unknown smoothing parameters.')
        if smoothing_method.startswith('inverse_smooth'):
            smooth_split = smoothing_method.split(',')
            if len(smooth_split) == 2:
                self.smoothing_method, self.smoothing_k = smooth_split
                self.smoothing_k = float(self.smoothing_k)
            elif len(smooth_split) == 1:
                self.smoothing_method = 'inverse_smooth'
                self.smoothing_k = 0.1
            else:
                raise ValueError('Unknown smoothing parameters.')
        elif smoothing_method.startswith('log'):
            log_split = smoothing_method.split(',')
            if len(log_split) == 2:
                self.smoothing_method, self.smoothing_target = log_split
                self.smoothing_target = float(self.smoothing_target)
                if self.smoothing_target <= 1:
                    raise ValueError('Log base must be > 1, current base = {}'.format(self.smoothing_target))
            elif len(log_split) == 1:
                self.smoothing_method = 'log'
                self.smoothing_target = 10
            else:
                raise ValueError('Unknown smoothing parameters.')
        elif smoothing_method.startswith('piecewise'):
            piecewise_split = smoothing_method.split(',')
            if len(piecewise_split) == 2:
                self.smoothing_method, self.smoothing_target = piecewise_split
                self.smoothing_target = float(self.smoothing_target)
                self.smoothing_k = 10
            elif len(piecewise_split) == 3:
                self.smoothing_method, self.smoothing_target, self.smoothing_k = piecewise_split
                self.smoothing_target = float(self.smoothing_target)
                self.smoothing_k = float(self.smoothing_k)
            elif len(piecewise_split) == 1:
                self.smoothing_method = self.smoothing_method
                self.smoothing_target = 20
                self.smoothing_k = 10
            else:
                raise ValueError('Unknown smoothing parameters. ')
        else:
            self.smoothing_method = smoothing_method

    @staticmethod
    def find_intersection_flatten(df, info_file):
        print('Searching intersecting values. ')
        with open(info_file, 'r') as fp:
            line = fp.readline()
            n_items = int(line.split(',')[1])
        df1 = df[:n_items]
        df2 = df[n_items:]
        #     Code to perform word-wise intersection
        # s1 = set([str(_) for word in df1.values.ravel().tolist() for _ in word.split('_')])
        # s2 = set([str(_) for word in df2.values.ravel().tolist() for _ in word.split('_')])
        print('Working on df1.')
        s1 = set([str(_) for _ in df1.values.ravel().tolist()])
        print('Working on df2.')
        s2 = set([str(_) for _ in df2.values.ravel().tolist()])

        intersection = s1.intersection(s2)

        return list(intersection)

    @staticmethod
    def evaluate_frequencies(flatten, df, intersection):
        if flatten and intersection:
            split_values = []
            for val in df.values.ravel().tolist():
                if val not in intersection:
                    try:
                        split = val.split('_')
                    except AttributeError:
                        split = [str(val)]
                else:
                    split = [str(val)]
                split_values += split
            frequencies = dict(Counter(split_values))
        elif flatten:
            split_values = []
            for val in df.values.ravel().tolist():
                try:
                    split = val.split('_')
                except AttributeError:
                    split = [str(val)]
                split_values += split
            frequencies = dict(Counter(split_values))
        else:
            frequencies = dict(Counter([str(_) for _ in df.values.ravel().tolist() if _ == _]))
        # Remove null values if they somehow slipped in.
        frequencies.pop('', None)
        frequencies.pop(np.nan, None)

        return frequencies

    @staticmethod
    def prepare_split(cell_value, flatten, intersection):
        if flatten and intersection:
            if cell_value in intersection:
                valsplit = [cell_value]
            else:
                valsplit = cell_value.split('_')
        elif flatten:
            valsplit = cell_value.split('_')
        else:
            valsplit = [cell_value]
        return valsplit

    def __init__(self, df, edgefile, prefixes, info_file=None, smoothing_method='no', flatten=False):
        """Data structure used to represent dataframe df as a graph. The data structure contains a list of all nodes
        in the graph, built according to the parameters passed to the function.

        :param df: dataframe to convert into graph
        :param sim_list: optional, list of pairs of similar values
        :param smoothing_method: one of {no, smooth, inverse_smooth, log, inverse}
        :param flatten: if set to True, spread multi-word tokens over multiple nodes. If set to false, all unique cell
        values will be merged in a single node.
        """
        self._parse_smoothing_method(smoothing_method)
        # df = df.fillna('')
        self.edgelist = []

        numeric_columns = []

        for col in df.columns:
            try:
                df[col].dropna(axis=0).astype(float).astype(str)
                numeric_columns.append(col)
            except ValueError:
                pass

        if info_file:
            intersection = self.find_intersection_flatten(df, info_file)
        else:
            intersection = []

        frequencies = self.evaluate_frequencies(flatten, df, intersection)

        count_rows = 1
        with open(edgefile, 'w', encoding='utf-8') as fp:
            fp.write(','.join(prefixes) + '\n')
            # Iterate over all rows in the df
            for idx, r in tqdm(df.iterrows()):
                rid = 'idx__' + str(idx)

                # Remove nans from the row
                row = r.dropna()
                # Create a node for the current row id.
                for col in df.columns:
                    try:
                        og_value = row[col]
                        cell_value = self.convert_cell_value(og_value)
                        # If cell value is None, continue
                        if not cell_value:
                            continue

                        # Tokenize cell_value depending on the chosen strategy.
                        valsplit = self.prepare_split(cell_value, flatten, intersection)

                        for split in valsplit:
                            try:
                                smoothed_f = self.smooth_freq(frequencies[split])
                            except KeyError:
                                smoothed_f = 1
                            n1 = rid
                            if col in numeric_columns:
                                n2 = 'tn__' + split
                            else:
                                n2 = 'tt__' + split

                            w1 = 1
                            w2 = smoothed_f
                            self.edgelist.append((n1, n2, w1, w2))
                            edgerow = '{},{},{},{}\n'.format(n1, n2, w1, w2)

                            fp.write(edgerow)
                            if col in numeric_columns:
                                n1 = 'tn__' + split
                            else:
                                n1 = 'tt__' + split

                            n2 = 'cid__' + col
                            w1 = smoothed_f
                            w2 = 1
                            self.edgelist.append((n1, n2, w1, w2))

                            edgerow = '{},{},{},{}\n'.format(n1, n2, w1, w2)

                            fp.write(edgerow)
                    except KeyError:
                        continue

                print('\r# {:0.1f} - {:}/{:} tuples'.format(count_rows / len(df) * 100, count_rows, len(df)), end='')
                count_rows += 1

        print('')

    def get_edgelist(self):
        return self.edgelist

    def convert_to_dict(self):
        self.graph_dict = {}
        for edge in self.edgelist:
            if len(edge) == 4:
                n1, n2, w1, w2 = edge
            elif len(edge) == 3:
                n1, n2, w1 = edge
                w2 = 0
            else:
                raise ValueError(f'Edge {edge} contains errors.')

            if n1 in self.graph_dict:
                self.graph_dict[n1][n2] = {'weight': w1}
            else:
                self.graph_dict[n1] = {n2: {'weight': w1}}

            if n2 in self.graph_dict:
                self.graph_dict[n2][n1] = {'weight': w2}
            else:
                self.graph_dict[n2] = {n1: {'weight': w2}}

        return self.graph_dict

    def convert_to_numeric(self):
        i2n = {idx: node_name for idx, node_name in enumerate(self.graph_dict.keys())}
        n2i = {node_name: idx for idx, node_name in i2n.items()}

        numeric_dict = {}

        for node in self.graph_dict:
            adj = self.graph_dict[node]
            new_adj = [n2i[_] for _ in adj]
            numeric_dict[n2i[node]] = new_adj

        return numeric_dict

def main(input_file, output_file, info_file=None, export=None):
    #args = parse_args()
    dfpath = input_file
    edgefile = output_file

    if info_file != None:
        info = info_file
        if tokenization == '':
            pass
    else:
        info = None

    df = pd.read_csv(dfpath, low_memory=False)

    pref = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']

    el = EdgeList(df, edgefile, pref, info, flatten=True)

    if export != None:
        el.convert_to_dict()
        gdict = el.convert_to_numeric()
        g_nx = nx.from_dict_of_lists(gdict)
        n, _ = osp.splitext(edgefile)
        nx_fname = n + '.nx'
        pkl_fname = n + '.pkl'
        pickle.dump(g_nx, open(nx_fname, 'wb'))
        pickle.dump(gdict, open(pkl_fname, 'wb'))

    # Loading the graph to make sure it can load the edgelist.
    # g = Graph(el.get_edgelist(), prefixes=pref)
