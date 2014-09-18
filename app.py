# -*- coding: utf-8 -*-
import StringIO


__author__ = 'developer'

import os
import web
import pickle
import base64
import random
from datetime import datetime
from PIL import Image


urls = (
    '/', 'Index',
    '/teach', 'Teach',
    '/(.+)', 'StaticFiles'
)


class Index:
    def GET(self):
        with open('static/index.html', 'rb') as rb:
            html = rb.read()
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        return html.replace('ip.gif', random.choice(os.listdir('static/images-learn')))

    def POST(self):
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        img_str = web.input(name=['image']).get('image', 'data:image/png;base64,')
        img_data = base64.b64decode(img_str.replace('data:image/png;base64,', ''))
        im = Image.open(StringIO.StringIO(img_data))
        data = [int(px != (0, 0, 0, 0)) for px in iter(im.getdata())]
        with open('trained.net', 'r') as file_object:
            net = pickle.load(file_object)
        # каталог с изображениями для обучения
        src = 'static/images-learn'
        files_known = os.listdir(src)
        # соответсвия между именем распозноваемого файла и цифрой на выходе из сети
        codes = dict()
        # цифра на выходе
        num = 0
        # перебираем все каталоги, в которых файлы для обучения
        for src_dir in files_known:
            if os.listdir(src + '/' + src_dir):
                codes[num] = src_dir
                # назачаем новое число каждому образу
                num += 1
        result = round(net.activate(data)[0])
        if result in codes:
            return str(codes) + u'<p>Распознано как</p><p><img src="images-src/%s" alt="Распознано как"></p>' % codes[result]
        return u'Не распознано'


class Teach:
    def GET(self):
        with open('static/index.html', 'rb') as rb:
            html = rb.read()
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        return html.replace('ip.gif', random.choice(os.listdir('static/images-learn')))

    def POST(self):
        web.header('Content-Type', 'text/html; charset=utf-8', unique=True)
        img_str = web.input(name=['image']).get('image', 'data:image/png;base64,')
        filename_src = web.input(name=['file']).get('file', None)
        if not filename_src:
            return u'Не сохранено'
        data = base64.b64decode(img_str.replace('data:image/png;base64,', ''))

        filename = 'static/images-learn/' + filename_src + '/'
        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = "_".join([filename, suffix])

        with open(filename + ".png", 'wb') as f:
            f.write(data)
        return u'Сохранено'


class StaticFiles():
    def GET(self, path):
        static_path = ''.join((os.path.abspath(os.path.curdir), "/static"))
        # print staticPath
        for d, dirs, files in os.walk(static_path):
            for f in files:
                #realtive path to dir
                directory = d.replace(static_path, '')
                #realtive path to file
                local_path = '/'.join([directory, f])[1:]
                #file is found
                if local_path == path:
                    file_path = '/'.join([static_path, local_path])
                    ext = file_path.split(".")[-1]
                    image_types = {
                        "png": "images/png",
                        "jpg": "images/jpeg",
                        "gif": "images/gif",
                        "ico": "images/x-icon"
                    }
                    if ext in image_types:
                        web.header("Content-Type", image_types[ext])
                    return open(file_path).read()

        #file is NOT found!
        raise web.notfound("404 Not found!")

    def POST(self, path):
        return self.GET(path)


# WSGIServer creating functions
def run_gevent_server(app, ip, port=8080):
    try:
        from gevent.pywsgi import WSGIServer

        WSGIServer((ip, port), app).serve_forever()
        return True
    except KeyboardInterrupt:  # If server stopped
        return True  # don't show KeyboardInterrupt exception
    except ImportError:  # If gevent module not found
        return False  # resurn that server not started


def run_simple_httpd_server(app, ip, port=8080):
    from wsgiref.simple_server import make_server

    make_server(ip, port, app).serve_forever()


if __name__ == '__main__':
    ip_adr = os.getenv('OPENSHIFT_PYTHON_IP') or '0.0.0.0'
    bind_port = 18080 if os.sys.platform == 'win32' else 8080
    # unsing web - web.py framework
    application = web.application(urls, globals()).wsgifunc()

    # Use gevent if we have it, otherwise run a simple httpd server.
    print 'Starting WSGIServer on %s:%d ... ' % (ip_adr, bind_port)
    if not run_gevent_server(application, ip_adr, bind_port):
        print 'gevent probably not installed - using default simple server ...'
        run_simple_httpd_server(application, ip_adr, bind_port)









