import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer, matthews_corrcoef
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


class ml_model:
    def __init__(self, file_path):
        self.file_path = file_path
        self.train_model()
        self.svm_model()
        self.logistic_regressor()
        self.linear_svm()
        

    #Function to one-hot encode the sequences
    def one_hot_encode(self, sequences):
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X', 'U', 'B', 'Z', 'O']
        
        encoded_sequences = []
        
        for sequence in sequences:
            encoded_sequence = []
            for amino_acid in sequence:
                vector = [0] * len(amino_acids)
                if amino_acid in amino_acids:
                    vector[amino_acids.index(amino_acid)] = 1
                encoded_sequence.append(vector)
            encoded_sequences.append(encoded_sequence)
        
        padded_sequences = pad_sequences(encoded_sequences, padding='post', dtype='float32')
        
        return padded_sequences



    def train_model(self):

        dataset = pd.read_csv(self.file_path, sep=',', header=0, index_col=0)

        df = dataset[dataset['sequence'].str.len() <= 500]

        protein_sequences = list(df['sequence'].values)

        encoded_sequences = self.one_hot_encode(protein_sequences)

        flattened_sequences = encoded_sequences.reshape(encoded_sequences.shape[0], -1)

        labels= df['label'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(flattened_sequences, labels, test_size=0.20, random_state=42, stratify=labels)


    def svm_model(self):

        clf = svm.SVC(class_weight='balanced', probability=True)
        clf.fit(self.X_train, self.y_train)
        pickle.dump(clf, open('/home/rfernandes/AllerGenProt/ml_model/svm_model.sav', 'wb'))
        predictions = clf.predict(self.X_test)
        score = classification_report(self.y_test, predictions)
        output_file = '/home/rfernandes/allerStat/ml_model/svm_output.txt'
        MCCSVM = matthews_corrcoef(self.y_test, predictions)
        
        with open(output_file, 'w') as f:
            f.write(f'SVM model:\n {score}\n MCC SVM: {MCCSVM}')

        return score

    def linear_svm(self):

        param_grid = {'C': [0.1, 1, 10, 100],
              'class_weight': ['balanced', None]}

        scorer = make_scorer(f1_score, pos_label=1)

        lclf = GridSearchCV(LinearSVC(class_weight='balanced'), param_grid=param_grid, verbose=3, cv=5, scoring=scorer)
        lclf.fit(self.X_train, self.y_train)
        pickle.dump(lclf, open('/home/rfernandes/AllerGenProt/ml_model/lclf_model.sav', 'wb'))
        predictions = lclf.predict(self.X_test)
        score = classification_report(self.y_test, predictions)
        MCCSVM = matthews_corrcoef(self.y_test, predictions)

        output_file = '/home/rfernandes/allerStat/ml_model/linear_svm_output.txt'

        with open(output_file, 'w') as f:
            f.write(f'Linear SVM model:\n {score}  \n MCC SVM: {MCCSVM}')

        return score

    def logistic_regressor(self):

        param_grid = {'C': [0.5, 1, 10, 100],
              'class_weight': ['balanced', None],
            }   

        scorer = make_scorer(f1_score, pos_label = 1)

        grid_lr = GridSearchCV(LogisticRegression(max_iter=10000),param_grid, verbose=3, cv=5, scoring=scorer)
        grid_lr.fit(self.X_train, self.y_train)
        pickle.dump(grid_lr, open('/home/rfernandes/AllerGenProt/ml_model/grid_lr_model.sav', 'wb'))
        y_pred_grid_lr = grid_lr.predict(self.X_test)
        score = classification_report(self.y_test, y_pred_grid_lr)
        MCCSVM = matthews_corrcoef(self.y_test, y_pred_grid_lr)
        print("MCC SVM:", MCCSVM)

        output_file = '/home/rfernandes/allerStat/ml_model/logistic_regressor_output.txt'

        with open(output_file, 'w') as f:
            f.write(f'Logistic regressor model:\n {score} \n MCC SVM: {MCCSVM}')

        return score


if __name__ == '__main__':
    model = ml_model('/home/rfernandes/allerStat/ml_model/prot_allergy_500.csv')
