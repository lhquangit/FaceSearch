import os

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.backends.cudnn as cudnn
import towhee
from tqdm.auto import tqdm

from .data import cfg_mnet
from .layers.functions.prior_box import PriorBox
from .models.retinaface import RetinaFace
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms

CONFIDENCE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.4
KEEP_TOP_K = 750


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #  print("Missing keys:{}".format(len(missing_keys)))
    #  print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    #  print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
    #  print("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    #  print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage
        )
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


torch.set_grad_enabled(False)
cfg = None
cfg = cfg_mnet


# net and model
net = RetinaFace(cfg=cfg, phase="test")
net = load_model(
    net, pretrained_path="weights/mobilenet0.25_Final.pth", load_to_cpu=True
)
net.eval()
#  print("Finished loading model!")
#  print(net)
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)


sess = ort.InferenceSession("weights/webface_r50.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


@towhee.register("extract_embedding")
def predict_face(img_raw):

    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape  # type: ignore
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])  # type: ignore
    img -= (104, 117, 123)  # type: ignore
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg["variance"])
    scale1 = torch.Tensor(
        [
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
            img.shape[3],
            img.shape[2],
        ]
    )
    scale1 = scale1.to(device)
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, NMS_THRESHOLD)

    dets = dets[keep, :]  # type: ignore
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:KEEP_TOP_K, :]

    dets = np.concatenate((dets, landms), axis=1)
    for k in range(dets.shape[0]):
        xmin = int(dets[k, 0] * im_width)
        ymin = int(dets[k, 1] * im_height)
        xmax = int(dets[k, 2] * im_width)
        ymax = int(dets[k, 3] * im_height)
        score = dets[k, 4]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
    face_cv = img_raw[ymin:ymax, xmin:xmax]  # type: ignore
    bbox = [xmin, ymin, xmax, ymax, score]  # type: ignore
    face_cv = cv2.resize(face_cv, (112, 112))
    face = (face_cv.transpose(2, 0, 1).astype(np.float32) - 127.5) / 255

    emb = sess.run([label_name], {input_name: np.array([face]).astype(np.float32)})[0]
    # return bbox, face_cv, emb
    return emb.squeeze()


# folder = "datasets/rikai/"
# embs = []
# ids = []


# for path in tqdm(os.listdir(folder)):
#     img_path = folder + path + "/" + os.listdir(folder + path)[0]

#     img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
#     try:
#         bbox, face, emb = predict_face(img_raw)
#         embs.append(emb[0])
#         ids.append(path)
#     except:
#         pass


# embs = np.array(embs)
# ids = np.array(ids)
# np.save("embs/rikai_embs", embs)
# np.save("embs/rikai_ids", ids)


# embs = np.load("embs/rikai_embs.npy")
# ids = np.load("embs/rikai_ids.npy")


# def face_search(img_raw):
#     bbox, face, emb = predict_face(img_raw)
#     cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
#     output = cosine_similarity(embs, emb.reshape(1, -1))

#     names_kq = ids[np.flip(np.argsort(output.reshape(-1)))[:3]]
#     scores_kq = output[np.flip(np.argsort(output.reshape(-1)))[:3]]
#     print(names_kq, scores_kq)
#     kq = []

#     for i, path in enumerate(names_kq):
#         img_path = folder + path + "/" + os.listdir(folder + path)[0]
#         img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
#         img_raw = cv2.resize(img_raw, (480, 640))
#         kq.append(img_raw)
#     return (names_kq[0], kq[0], names_kq[1], kq[1], names_kq[2], kq[2])
