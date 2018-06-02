# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:12:48 2018

@author: ASUS
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import time

MAX_LIMIT = 9999999
MIN_LIMIT = -9999999




class Hypothesis():
    def __init__(self, attr, preds, avg_cl=None):
        self.attr_name = attr
        self.predictions = preds
        self.avg_cl = avg_cl
    def get_hypo(self):
        return (self.attr_name, self.predictions)

    def get_avg(self):
        return self.avg_cl

    def get_pred(self, value):

        classes = list(self.predictions.keys())
#        value = self.data[attr_name].values[idx]



        class_key = None
        if type(value) == str:
            for p in classes:
                if p == value:
                    class_key = p
                    break
            if class_key == None:
                r = np.random.randint(0, 9999)%2
                if r == 1:
                    return 1
                else:
                    return -1
        else:
            if value <= self.avg_cl:
                class_key = 'less'
            else:
                class_key = 'not_less'

        count_yes, count_no = self.predictions[class_key]

        if count_yes >= count_no:
            return 1
        else:
            return -1

    def print_h(self):
        print(self.predictions)



class DecisionStud():
    def __init__(self, data, w):
        self.data = data
        self.data_columns = list(data.columns)
        self.entropy = []
        self.weights = w

    def return_type(self, name):
        return type(self.data[name].values[0])

    def get_label_count_str(self, attr_name, attrValue, attrLabel):


        count_yes = 0
        count_no = 0
        datam = self.data.loc[self.data[attr_name] == attrValue ]
        data_yes = datam.loc[datam[attrLabel] == 'yes']
        if len(data_yes) !=0:
            count_yes = len(data_yes)


        data_no = datam.loc[datam[attrLabel] == 'no']
        if len(data_no) != 0:
            count_no = len(data_no)

        return (count_yes, count_no)

    def get_label_count_num(self, attr_name, prevValue, attrValue, attrLabel):

        if prevValue == 'less':
            datam = self.data.loc[self.data[attr_name] <= attrValue ]
            data_yes = datam.loc[datam[attrLabel] == 'yes']
            count_yes = len(data_yes)


            data_no = datam.loc[datam[attrLabel] == 'no']
            count_no = len(data_no)

        else:
            datam = self.data.loc[self.data[attr_name] > attrValue ]
            data_yes = datam.loc[datam[attrLabel] == 'yes']
            count_yes = len(data_yes)


            data_no = datam.loc[datam[attrLabel] == 'no']
            count_no = len(data_no)

        return (count_yes, count_no)


    def calc_entropy(self, yes, no):
        t = yes+no
        yes_val=0
        no_val=0
        if yes != 0:
            yes_val = -(yes/t)*np.log2(yes/t)
        if no != 0:
            no_val = -(no/t)*np.log2(no/t)
        return yes_val+no_val

    def get_numeric_entropy(self, attr_name):
        attr_data = self.data[attr_name].values
        entropy_list = []
        data_count = []
        prev = 'less'

        avg_cl = np.mean(attr_data)

        count_yes, count_no = self.get_label_count_num(attr_name,prev, avg_cl, 'y')
        if count_yes + count_no == 0:
            entropy = 0
        else:
            entropy = self.calc_entropy(count_yes, count_no)
        entropy_list.append(entropy)
        data_count.append(count_no + count_yes)

        prev = 'not_less'

        count_yes, count_no = self.get_label_count_num(attr_name,prev, avg_cl, 'y')
        if count_yes + count_no == 0:
            entropy = 0
        else:
            entropy = self.calc_entropy(count_yes, count_no)
        entropy_list.append(entropy)
        data_count.append(count_no + count_yes)
        sum_t = np.sum(data_count)
        total_entropy = 0
        for val_idx in range(len(entropy_list)):
            total_entropy += (data_count[val_idx]/sum_t)*entropy_list[val_idx]
        return total_entropy

    def get_str_entropy(self, attr_name):
        attr_data = self.data[attr_name].values
        attr1_data = np.unique(attr_data)
        entropy_list = []
        data_count = []
        for val in attr1_data:
            count_yes, count_no = self.get_label_count_str(attr_name, val, 'y')

            if count_yes + count_no == 0:
                entropy = 0
            else:
                entropy = self.calc_entropy(count_yes, count_no)

            entropy_list.append(entropy)
            data_count.append(count_no + count_yes)

        sum_t = np.sum(data_count)
        total_entropy = 0
        for val_idx in range(len(entropy_list)):
            total_entropy += (data_count[val_idx]/sum_t)*entropy_list[val_idx]
        return total_entropy


    def get_entropy(self, attr_name):
        typ = self.return_type(attr_name)
        if  typ == str:
            #print("str: ",attr_name)
            return self.get_str_entropy(attr_name)
        else:
            #print("number: ",attr_name)
            return self.get_numeric_entropy(attr_name)



    def populate_entropy_list(self):

        for attr_name in self.data_columns:
            if attr_name == 'y':
                continue
            entr = self.get_entropy(attr_name)
            self.entropy.append(entr)

    def fit(self):
        self.populate_entropy_list()
#        print(self.entropy)
        attr_idx = np.argmin(self.entropy)
#        print(attr_idx)
        attr_name = self.data_columns[attr_idx]
#        print(attr_name)
        dataValues1 = self.data[attr_name].values
        dataValues = np.unique(dataValues1)

        attr_dict = {}
#        prev = MIN_LIMIT

        typ = self.return_type(attr_name)
        if  typ == str:
            for i in range(len(dataValues)):


                    count_yes, count_no = self.get_label_count_str(attr_name, dataValues[i], 'y')
                    attr_dict[dataValues[i]] = (count_yes, count_no)
            hyp = Hypothesis(attr_name, attr_dict)
        else:
#            min_cl = np.min(dataValues)
#            max_cl = np.max(dataValues)
            avg_cl = np.mean(dataValues1)

            count_yes, count_no = self.get_label_count_num(attr_name, 'less', avg_cl, 'y')
            attr_dict['less'] = (count_yes, count_no)

            count_yes, count_no = self.get_label_count_num(attr_name, 'not_less', avg_cl, 'y')
            attr_dict['not_less'] = (count_yes, count_no)

            hyp = Hypothesis(attr_name, attr_dict, avg_cl)


        return hyp

class Adaboost():

    def __init__(self, data, k):
        self.data = data
        self.K = k
        self.data_columns = list(data.columns)
        self.N = len(data)
        self.weights = [1/self.N]*self.N
        self.hypotheses = []
        self.hypo_weights = []
        self.cum_weights = [0]*self.N



    def make_cum_weights(self):
        sum_c = 0
        for idx in range(self.N):
            sum_c += self.weights[idx]
            self.cum_weights[idx] = sum_c

    def resample(self):
        #print("resampling start")
        #start_time = time.clock()
        #self.make_cum_weights()
        resample_data = pd.DataFrame(columns=self.data_columns)

        #for i in range(self.N):
        float_val = np.random.uniform(0, 1, self.N)

        arr_w = np.cumsum(self.weights)

        arr_idx = np.searchsorted(arr_w, float_val)

        resample_data = resample_data.append(self.data.iloc[arr_idx].copy(), ignore_index=True)
            #resample_data = shuffle(resample_data)

        #print("resampling stop")
        #end_time = time.clock()

        #print(end_time-start_time)

        return resample_data

    def test_hypo(self, hypo, idx):
        attr_name, preds = hypo.get_hypo()
        classes = list(preds.keys())
#        values = self.data[attr_name].values
        value = self.data[attr_name].values[idx]
        outcome = self.data['y'].values[idx]

        class_key = None

        if type(value) == str:
            for p in classes:
                if p == value:
                    class_key = p
                    break
            if class_key == None:
                r = np.random.randint(0, 9999)%2

                if r == 0:
                    prophecy = 'no'
                else:
                    prophecy = 'yes'

                return prophecy == outcome
        else:

            avg_cl = hypo.get_avg()
            if value <= avg_cl:
                class_key = 'less'
            else:
                class_key = 'not_less'

        count_yes, count_no = preds[class_key]
        prediction = ''
        if count_yes >= count_no:
            prediction = 'yes'
        else:
            prediction = 'no'
        return prediction == outcome



    def normalize_weights(self):
        sum_w = np.sum(self.weights)
        for i in range(self.N):
            self.weights[i] /= sum_w

    def weighted_decision(self, series):
        prediction = 0
        for i in range(len(self.hypotheses)):
            attr_name, preds = self.hypotheses[i].get_hypo()
            value = series[attr_name]
            prediction += self.hypotheses[i].get_pred(value)*self.hypo_weights[i]
        return prediction





    def boost(self):
        k = self.K
        while(k>0):
            resampled_data = self.resample()
            learner = DecisionStud(resampled_data, self.weights)
            hypo = learner.fit()

            error = 0


            for j in range(self.N):
                if not self.test_hypo(hypo, j):
                    error += self.weights[j]


            if error > 0.5:
                print("error: ",error)
                continue
            for j in range(self.N):
                if self.test_hypo(hypo, j):
                    self.weights[j] *= (error/(1-error))
            k-=1
            self.normalize_weights()
            self.hypotheses.append(hypo)
            self.hypo_weights.append(np.log2((1-error)/error))
        return (self.hypotheses, self.hypo_weights)



def data_preprocess(filename, data_columns):
    data = pd.read_csv(filename, sep=';', encoding='utf-8')
    data.columns = data_columns
    data_yes = data.loc[data['y'] == 'yes']


    data_no = data.loc[data['y'] == 'no']
    data_no = shuffle(data_no, random_state=10)

    data_no = data_no.iloc[0:len(data_yes)].copy()

    data_comb = data_yes.append(data_no)
    data_comb = shuffle(data_comb, random_state=10)
    print(len(data_comb))


    return data_comb


def main():
    filename = 'F:/Class materials/L4T2/ML/offline1/bank-full.csv'
    columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
    data = data_preprocess(filename, columns)
    folds =5
    kf = KFold(n_splits = folds, shuffle = True, random_state = 20)
    f1_score = 0
    fold_f1 = []
    np.random.seed(10)
    for k in range(folds):
        result = next(kf.split(data), None)

        data_train = data.iloc[result[0]]
        data_test = data.iloc[result[1]]
        adaboost = Adaboost(data_train, 1)
        hypotheses, weights = adaboost.boost()
        print('Fold ',k,':')
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(data_test)):
            predValue = adaboost.weighted_decision(data_test.iloc[i])
            if predValue >= 0:
                pred = 'yes'
            else:
                pred = 'no'
            correct = data_test['y'].iloc[i]
            if pred == correct:
                if pred == 'yes':
                    TP += 1
                else:
                    TN += 1
            else:
                if pred == 'yes':
                    FP += 1
                else:
                    FN += 1

        print('TP: ',TP)
        print('TN: ',TN)
        print('FP: ',FP)
        print('FN: ',FN)
        prec = TP/(TP+FP)
        recall = TP/(TP+FN)
        f = (2*prec*recall)/(prec+recall)
        print('Precision: ',prec)
        print('Recall: ', recall)
        print('Fold ',k,': F1 Score: ',f)
        f1_score += f
        fold_f1.append(f)
    print('Average F1 Score: ',f1_score/folds)
    print(fold_f1)

if __name__ == "__main__":
    main()
