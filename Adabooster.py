import numpy as np


class Adabooster:
    def __init__(self, training_set):
        self.training_set = training_set
        self.training_size = len(self.training_set)
        self.weights = np.ones(self.training_size) / self.training_size
        self.weak_learners = []
        self.weak_alphas = []

    def __evaluate_errors(self, results):
        # TODO
        return results

    def __update_weights(self, errors):
        e = (errors*self.weights).sum()
        alpha = 0.5 * np.log((1-e)/e)
        w = np.zeros(self.training_size)
        for i in range(self.training_size):
            if errors[i] == 1:
                w[i] = self.weights[i] * np.exp(alpha)
            else:
                w[i] = self.weights[i] * np.exp(-alpha)
        self.weights = w / w.sum()
        self.weak_alphas.append(alpha)

    def set_weak_learner(self, weak_learner):
        results = np.array(weak_learner.train(t) for t in self.training_set)
        errors = self.__evaluate_errors(results=results)
        self.__update_weights(errors=errors)
        self.weak_learners.append(weak_learner)

    def evaluate_model(self):
        weak_learners_len = len(self.weak_learners)
        for x in self.training_set:
            hx = [self.weak_alphas[i]*self.weak_learners[i](x) for i in range(weak_learners_len)]
            # TODO - Decide on evaluation function

    def predict_data(self, test_data):
        return sum(weight * learner(test_data) for (weight, learner) in zip(self.weights, self.weak_learners))