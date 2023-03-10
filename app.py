import os

from flask import Flask, request

from commonAPI import template_matching, response_format, feature_matching

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/upload', methods=['POST'])
def upload():
    BASE_PIC_DIR = '/Users/admin/PycharmProjects/imgMatchServer/'
    picNameArray = ['template.png', 'target.png']

    if request.method == 'GET':
        return "请求方式错误"
    elif request.method == "POST":
        m_dict = request.files
        k = 0
        """
        遍历文件列表，读取文件存放本地
        """
        for i in m_dict.values():
            i.name = picNameArray[k]
            with open(os.path.join(os.getcwd(), i.name), 'wb') as fw:
                # 一次性读取文件
                fw.write(i.read())
                # 分块读取文件
                # for w in i.chunks():
                #     fw.write(w)
            k = k + 1
        """
        调用特征点匹配和模板匹配方法，寻找目标元素中心坐标
        """
        try:
            fea_pos = feature_matching(BASE_PIC_DIR + picNameArray[0], BASE_PIC_DIR + picNameArray[1])
            if fea_pos is not None:
                response = response_format(0, "成功", fea_pos)
                return response
            else:
                fea_pos = template_matching(BASE_PIC_DIR + picNameArray[0], BASE_PIC_DIR + picNameArray[1])
        except BaseException as err:
            response = response_format(3001, "图像匹配异常", fea_pos)
            return response


if __name__ == '__main__':
    app.run()


