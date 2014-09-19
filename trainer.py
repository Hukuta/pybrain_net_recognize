#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pybrain.utilities import fListToString
from pybrain.supervised.trainers import BackpropTrainer


class Trainer(BackpropTrainer):
    def trainUntilConvergence(self, dataset=None, maxEpochs=None, verbose=None, continueEpochs=10,
                              validationProportion=0.25, maxError=None):
        """Train the module on the dataset until it converges.

        Return the module with the parameters that gave the minimal validation
        error.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. validationProportion is the ratio of the dataset
        that is used for the validation dataset.

        If maxEpochs is given, at most that many epochs
        are trained. Each time validation error hits a minimum, try for
        continueEpochs epochs to find a better one."""
        epochs = 0
        if dataset == None:
            dataset = self.ds
        if verbose == None:
            verbose = self.verbose
        # Split the dataset randomly: validationProportion of the samples for
        # validation.
        trainingData, validationData = (
            dataset.splitWithProportion(1 - validationProportion))
        if not (len(trainingData) > 0 and len(validationData)):
            raise ValueError("Provided dataset too small to be split into training " +
                             "and validation sets with proportion " + str(validationProportion))
        self.ds = trainingData
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(validationData)
        trainingErrors = []
        validationErrors = [bestverr]
        while True:
            trainingErrors.append(self.train())
            validationErrors.append(self.testOnData(validationData))
            if epochs == 0 or validationErrors[-1] < bestverr:
                # one update is always done
                bestverr = validationErrors[-1]
                bestweights = self.module.params.copy()

            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1

            if len(validationErrors) >= continueEpochs * 2:
                # have the validation errors started going up again?
                # compare the average of the last few to the previous few
                old = validationErrors[-continueEpochs * 2:-continueEpochs]
                new = validationErrors[-continueEpochs:]
                if min(new) > max(old):
                    self.module.params[:] = bestweights
                    break
            # остановка при достижении заданной точности
            if not maxError is None and trainingErrors[-1] <= maxError:
                self.module.params[:] = bestweights
                break
        trainingErrors.append(self.testOnData(trainingData))
        self.ds = dataset
        if verbose:
            print 'train-errors:', fListToString(trainingErrors, 6)
            print 'valid-errors:', fListToString(validationErrors, 6)
        return trainingErrors, validationErrors