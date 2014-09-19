#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

__author__ = 'Odarchenko N.D.'

import random
import pickle
import os
from PIL import Image
from pybrain.tools.shortcuts import buildNetwork

from pybrain.datasets import ClassificationDataSet
from trainer import Trainer


if __name__ == '__main__':
    start_time = time.time()
    # каталог с изображениями для обучения
    src = 'static/images-learn'
    files_known = os.listdir(src)

    # размер изображений 100*100px
    ds_sz = 100 * 100
    # список для обучения
    data_model = list()

    # цифра на выходе
    num = 0
    # перебираем все каталоги, в которых файлы для обучения
    for src_dir in sorted(files_known):
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

