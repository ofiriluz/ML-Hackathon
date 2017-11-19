import numpy as np


class Adabooster:
    def __init__(self, training_set):
        self.training_set = training_set
        self.training_size = len(self.training_set)
        self.validation_size = int(len(self.training_set)*0.2)
        self.weights = np.ones(self.validation_size) / self.validation_size
        self.weak_learners = []
        self.weak_alphas = []

    def __update_weights(self, errors):
        e = (errors*self.weights).sum()
        alpha = 0.5 * np.log((1-e)/e)
        w = np.zeros(int(len(self.training_set)*0.2))
        for i in range(int(len(self.training_set)*0.2)):
            w[i] = self.weights[i] * np.exp(alpha)
        self.weights = w / w.sum()
        self.weak_alphas.append(alpha)

    def set_weak_learner(self, weak_learner):
        weak_learner.reset()
        (mse, predictions) = weak_learner.train(self.training_set)
        # print(mse)
        # print(predictions)
        # results = np.array([weak_learner.train(t) for t in self.training_set])
        self.__update_weights(errors=mse)
        self.weak_learners.append(weak_learner)

    def evaluate_model(self):
        pass
        # weak_learners_len = len(self.weak_learners)
        # for x in self.training_set:
        #     hx = [self.weak_alphas[i]*self.weak_learners[i] for i in range(weak_learners_len)]
        #     TODO - Decide on evaluation function

    def predict_data(self, test_data):
        return self.weak_learners[0].predict(test_data)
        # return sum(weight * learner.predict(test_data)[0] for (weight, learner) in zip(self.weak_alphas, self.weak_learners))

    def save_model(self, folder_path):
        for (index, learner) in enumerate(self.weak_learners):
            learner.save_model(folder_path + '/' + 'model_' + str(index) + 'hd5')