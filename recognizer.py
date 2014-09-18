#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

__author__ = 'Odarchenko N.D.'

import random
import pickle
import os
from PIL import Image
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import fListToString


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


def teach():
    start_time = time.time()
    # каталог с изображениями для обучения
    src = 'static/images-learn'
    files_known = os.listdir(src)
    files_known.sort()

    # размер изображений 100*100px
    ds_sz = 100 * 100
    # список для обучения
    data_model = list()

    # цифра на выходе
    num = 0
    # перебираем все каталоги, в которых файлы для обучения
    for src_dir in files_known:
        if os.listdir(src + '/' + src_dir):
            # считываем все файлы, по которым учим сеть
            for file_teach in os.listdir(src + '/' + src_dir):
                im = Image.open(os.path.join(src, src_dir, file_teach))
                data = [int(px != (0, 0, 0, 0)) for px in iter(im.getdata())]
                # добавляем данные для обучения в виде списка
                data_model.append([data, num])
            # назачаем новое число каждому образу
            num += 1

    ds = ClassificationDataSet(ds_sz, nb_classes=num)
    ds_test = ClassificationDataSet(ds_sz, nb_classes=num)

    for inp, out in data_model:
        ds.appendLinked(inp, [out])
        ds_test.appendLinked(inp, [out])

    # добавим больше данных для обучения
    for n in xrange(800):
        for i in range(num):
            out = -1
            inp = None
            while out != i:
                inp, out = random.choice(data_model)
            ds.appendLinked(inp, [out])

    net = buildNetwork(ds_sz, 2, 1)
    trainer = Trainer(net, momentum=0.1, verbose=True, weightdecay=0.01)
    trainer.setData(ds)
    trainer.trainUntilConvergence(verbose=True, maxError=0.025)
    trainer.setData(ds_test)
    trainer.testOnData(verbose=True)

    print "--- %s seconds ---" % (int(time.time()) - int(start_time))

    with open('trained.net', 'w') as fileObject:
        pickle.dump(net, fileObject)


if __name__ == '__main__':
    teach()

