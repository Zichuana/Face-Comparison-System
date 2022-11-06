from flask import Flask, jsonify, render_template, request
from datetime import timedelta
import cv2
import io
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

app = Flask(__name__, static_url_path="/")

# 自动重载模板文件
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
# 设置静态文件缓存过期时间
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 获得人脸特征向量
def load_known_faces(dstImgPath, mtcnn, resnet):
    aligned = []
    knownImg = cv2.imread(dstImgPath)  # 读取图片
    face = mtcnn(knownImg)  # 使用mtcnn检测人脸，返回【人脸数组】
    print(face)
    if face is not None:
        aligned.append(face[0])
    print(aligned)
    aligned = torch.stack(aligned).to(device)
    with torch.no_grad():
        known_faces_emb = resnet(aligned).detach().cpu()  # 使用resnet模型获取人脸对应的特征向量
    # print("\n人脸对应的特征向量为：\n", known_faces_emb)
    return known_faces_emb, knownImg


# 计算人脸特征向量间的欧氏距离，设置阈值，判断是否为同一个人脸
def match_faces(faces_emb, known_faces_emb, threshold):
    isExistDst = False
    distance = (known_faces_emb[0] - faces_emb[0]).norm().item()
    if (distance < threshold):
        isExistDst = True
    return distance, isExistDst


@app.route('/', methods=['GET', 'POST'])
def root():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
@torch.no_grad()
def predict():
    info = {}
    try:
        image1 = request.files["file0"]
        image2 = request.files["file1"]
        img_bytes1, img_bytes2 = image1.read(), image2.read()
        image1, image2 = Image.open(io.BytesIO(img_bytes1)), Image.open(io.BytesIO(img_bytes2))
        image_path1, image_path2 = './data/a.png', './data/b.png'
        image1.save(image_path1)
        image2.save(image_path2)
        mtcnn = MTCNN(min_face_size=12, thresholds=[0.2, 0.2, 0.3], keep_all=True, device=device)
        # InceptionResnetV1模型加载【用于获取人脸特征向量】
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        MatchThreshold = 0.8  # 人脸特征向量匹配阈值设置

        known_faces_emb, _ = load_known_faces(image_path1, mtcnn, resnet)  # 已知人物图
        # bFaceThin.png  lyf2.jpg
        faces_emb, img = load_known_faces(image_path2, mtcnn, resnet)  # 待检测人物图
        distance, isExistDst = match_faces(faces_emb, known_faces_emb, MatchThreshold)  # 人脸匹配
        info["oushi"] = "两张人脸的欧式距离为：{}".format(distance)
        info["fazhi"] = "设置的人脸特征向量匹配阈值为：{}".format(MatchThreshold)
        print("OK")
        if isExistDst:
            boxes, prob, landmarks = mtcnn.detect(img, landmarks=True)  # 返回人脸框，概率，5个人脸关键点
            info["result"] = '由于欧氏距离小于匹配阈值，匹配！该判断方式下是一个人！'
        else:
            info["result"] = '由于欧氏距离大于匹配阈值，不匹配！该判断方式下不是一个人！'
    except Exception as e:
        info["err"] = str(e)
    return jsonify(info)  # json格式传至前端


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1234)
