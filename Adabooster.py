import numpy as np
import json


class Adabooster:
    def __init__(self, training_set, weak_learners):
        self.training_set = training_set
        self.weak_learners = weak_learners
        self.weak_alphas = []
        if training_set:
            self.training_size = len(self.training_set)
            self.validation_size = int(len(self.training_set)*0.2)
            self.weights = np.ones(self.validation_size) / self.validation_size
        else:
            self.training_size = self.validation_size = 0
            self.weights = np.zeros(shape=(0, 0))

    def __update_weights(self, errors):
        e = (errors*self.weights).sum()
        alpha = 0.5 * np.log((1-e)/e)
        w = np.zeros(self.validation_size)
        for i in range(self.validation_size):
            w[i] = self.weights[i] * np.exp(alpha)
        self.weights = w / w.sum()
        self.weak_alphas.append(alpha)

    def set_weak_learner(self, weak_learner, train=True):
        if train:
            weak_learner.reset()
            (mse, predictions) = weak_learner.train(self.training_set)
            self.__update_weights(errors=mse)
        self.weak_learners.append(weak_learner)

    def train(self):
        for (index, learner) in enumerate(self.weak_learners):
            training_dataset = [item[index].get_features() for item in self.training_set]
            learner.reset()
            (mse, predictions) = learner.train(training_dataset)
            self.__update_weights(errors=mse)
        print('Alphas = ' + str(self.weak_alphas))

    def predict(self, test_data):
        mses = []
        predictions = []
        for (index, learner) in enumerate(self.weak_learners):
            (mse, pred) = learner.predict(test_data[index].get_features())
            mses.append(mse)
            predictions.append(pred)
        final_mse = sum([mse[0] * self.weak_alphas[index] for (index, mse) in enumerate(mses)])
        return final_mse, predictions

    def save_booster(self, folder_path, prefix):
        metadata = {'weights': self.weights.tolist(), 'alphas': self.weak_alphas,
                    'training_size': self.training_size, 'validation_size': self.validation_size,
                    'models': []}
        for (index, learner) in enumerate(self.weak_learners):
            metadata['models'].append({'path': folder_path + '/' + prefix + '_model_' + str(index) + '.hd5',
               'metadata': learner.save_model(folder_path + '/' + prefix + '_model_' + str(index) + '.hd5')
                                       })
        return metadata

    def load_booster(self, metadata):
        self.weights = metadata['weights']
        self.weak_alphas = metadata['alphas']
        self.training_size = metadata['training_size']
        self.validation_size = metadata['validation_size']
        for (index, model_metadata) in enumerate(metadata['models']):
            self.weak_learners[index].load_model(model_metadata['path'], model_metadata['metadata'])
