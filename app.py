import os
# TCP cho RTSP: tranh mat goi UDP gay vo hinh/nhay frame. Phai dat truoc khi import cv2.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import json
import math
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from urllib.parse import quote

from flask import Flask, Response, jsonify, redirect, render_template
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont   # ve chu tieng Viet co dau (cv2.putText khong ho tro)
except ImportError:                               # thieu Pillow -> tu dong dung chu khong dau
    np = Image = ImageDraw = ImageFont = None
try:
    from scipy.optimize import linear_sum_assignment   # di kem Ultralytics
except ImportError:
    linear_sum_assignment = None
from ultralytics import YOLO

app = Flask(__name__)


def _env_bool(name, default):
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_int_list(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return [int(x) for x in v.split(",") if x.strip()]
    except ValueError:
        return default


def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:   # console Windows cu khong in duoc tieng Viet co dau
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


# ==================================================
# 1. CAU HINH
# ==================================================
MODEL_PATH = Path(__file__).resolve().parent / "best.pt"

CAMERA_IP = os.getenv("CAMERA_IP", "192.168.1.233")
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", "admin")
# KHONG hard-code mat khau trong source. Dat bang: setx CAMERA_PASSWORD "..."  (roi mo lai terminal)
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "Ngan@nhphat2026")
RTSP_PORT = int(os.getenv("RTSP_PORT", "554"))
RTSP_URL = (
    f"rtsp://{quote(CAMERA_USERNAME, safe='')}:{quote(CAMERA_PASSWORD, safe='')}"
    f"@{CAMERA_IP}:{RTSP_PORT}/Streaming/Channels/101"
)
RTSP_RECONNECT_DELAY = 2.0
PREVIEW_FPS = 20.0

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "5001"))  # 5000 da dung cho du an "camera"

PROFILE = os.getenv("COUNT_PROFILE", "v3").strip().lower()
DEBUG = _env_bool("COUNT_DEBUG", False)

CONFIDENCE = 0.5                      # nguong "da xac nhan" tu dong
CONF_LOW = float(os.getenv("CONF_LOW", "0.30"))   # nguong suy luan cua profile v2/v3
RECOVER_MIN_OBS = 3                   # track conf thap can xuat hien >= 3/5 frame
RECOVER_MIN_MEAN = 0.40               # va conf trung binh >= 0.40
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "1600"))
USE_PREPROCESS = _env_bool("USE_PREPROCESS", True)
AGNOSTIC_NMS = _env_bool("AGNOSTIC_NMS", False)
NMS_IOU = float(os.getenv("NMS_IOU", "0.5"))       # chi co tac dung khi model chay che do NMS
SCALES = _env_int_list("SCALES", [1280, 1600, 1920, 1600, 1280])   # v3: imgsz moi frame
REFINE = _env_bool("REFINE", True)

USE_ROI = True
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 0.05, 0.10, 0.95, 0.85

USE_SHAPE_CHECK = True
SHAPE_DEBUG = True
SHAPE_CORRECT = _env_bool("SHAPE_CORRECT", False)  # chi ap dung cho profile v2/v3
SHAPE_MIN_SIZE = 18
DEDUP_IOU = 0.65                                   # dedup cu (baseline / v2)

# ---- v3: geometry-aware dedup (can hieu chinh theo du lieu that) ----
DUP_IOU_SAME = 0.85     # cung class: chi coi la trung khi IoU >= nguong nay ...
DUP_CDIST = 0.08        # ... VA khoang cach tam / duong cheo nho nhat <= nguong nay ...
DUP_SIZE = 0.80         # ... VA ti le dien tich nho/lon >= nguong nay
DUP_IOU_CROSS = 0.80    # khac class cung 1 vat (vd nut/washer nham nhau)
FILL_X_MAX = 0.45       # minAreaRect fill < nguong -> hinh chu X / nhieu vat (phu quyet dedup)

# ---- v3: ghep frame bang Hungarian ----
MATCH_MAX_COST = 0.90
_BIG = 1e6

# ---- v3: re-inference vung nghi ngo ----
REFINE_IMGSZ = 960
MAX_REFINE = 6
SUSPECT_AREA_K = 1.6    # box screw > 1.6 x trung vi dien tich screw cua phien -> nghi gop

COUNTDOWN_SECONDS = 3
SAMPLE_COUNT = 5
SAMPLE_INTERVAL = 0.15
IOU_THRESHOLD = 0.30

REVIEW_CONFIDENCE = 0.70
REVIEW_COUNT_SPREAD = 2
REVIEW_ON_SHAPE_DISAGREEMENT = True
CLASS_SHARE_REVIEW = 0.80

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))
lock = Lock()


def detect_e2e_kwargs():
    """Tham so bat nhanh one-to-one (NMS-free) cua YOLO26 - tuy phien ban Ultralytics:
    ban moi: nms=False ; ban cu: end2end=True. Tra {} neu khong nhan dien duoc."""
    try:
        import numpy as _np
        dummy = _np.zeros((64, 64, 3), _np.uint8)
    except Exception:
        return {}
    for kw in ({"nms": False}, {"end2end": True}):
        try:
            model.predict(dummy, imgsz=64, verbose=False, **kw)
            return kw
        except Exception:
            continue
    return {}


E2E_KW = detect_e2e_kwargs()

camera_enabled = False
camera_ready = False
camera_thread = None
camera_status = "Đã dừng"

session_state = "idle"
session_message = "Sẵn sàng đếm"
session_id = 0
countdown_deadline = 0.0

latest_frame = None
latest_jpeg = None

latest_counts = {}
latest_total = 0
latest_detections = 0
latest_recovered = 0
latest_result_jpeg = None
latest_result_filename = None
latest_review_frame = None
latest_review_metadata = None
latest_review_reasons = []
latest_report = None

sample_frames = []
last_sample_time = 0.0


def make_cfg(profile=None, **override):
    """baseline = hanh vi cu; v2 = nguong thap + xac nhan theo track;
    v3 = v2 + geo-dedup + NMS-free + da scale + Hungarian + refine."""
    profile = (profile or PROFILE).lower()
    cfg = dict(name=profile, conf_infer=CONFIDENCE, recover=False,
               shape_correct=True, legacy_merge=True,
               preprocess=USE_PREPROCESS, imgsz=IMAGE_SIZE, agnostic=AGNOSTIC_NMS,
               nms_iou=NMS_IOU, geo=False, nms_free=False, scales=None,
               hungarian=False, refine=False)
    if profile != "baseline":
        cfg.update(conf_infer=min(CONFIDENCE, CONF_LOW), recover=True,
                   shape_correct=SHAPE_CORRECT, legacy_merge=False)
    if profile == "v3":
        cfg.update(geo=True, nms_free=True, scales=(SCALES or None),
                   hungarian=True, refine=REFINE)
    cfg.update(override)
    return cfg


def cname(class_id):
    return str(model.names[class_id]).lower()


def brief(d):
    return {"class": cname(d["class_id"]), "conf": round(d["confidence"], 3),
            "box": [int(round(v)) for v in d["box"]]}


# ==================================================
# 2. XU LY ANH
# ==================================================
def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def calculate_sharpness(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h * .15):int(h * .85), int(w * .15):int(w * .85)]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frame(frames):
    scores = [calculate_sharpness(f) for f in frames]
    idx = max(range(len(scores)), key=lambda i: scores[i])
    log(f"[FRAME SELECT] scores={[round(v, 2) for v in scores]} selected={idx + 1}/{len(frames)}")
    return frames[idx], idx, scores


def get_detections(result):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        detections.append({
            "box": [float(v) for v in box.xyxy[0].cpu().numpy().tolist()],
            "class_id": int(box.cls[0].item()),
            "confidence": float(box.conf[0].item()),
        })
    return detections


def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
    improved = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(improved, (0, 0), 0.7)
    return cv2.addWeighted(improved, 1.25, blur, -0.25, 0)


def get_roi_px(w, h):
    """Mot ham duy nhat cho ROI: dung chung cho inference, preview va bao cao."""
    if not USE_ROI:
        return 0, 0, w, h
    x1 = max(0, min(w - 1, int(round(w * ROI_X1))))
    y1 = max(0, min(h - 1, int(round(h * ROI_Y1))))
    x2 = max(x1 + 1, min(w, int(round(w * ROI_X2))))
    y2 = max(y1 + 1, min(h, int(round(h * ROI_Y2))))
    return x1, y1, x2, y2


def deduplicate(detections, threshold=DEDUP_IOU, log_list=None, stage="dedup"):
    """(baseline/v2) Greedy NMS khong phan biet class, ghi lai box bi loai + ly do."""
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for item in ordered:
        clash = None
        for old in kept:
            iou = calculate_iou(item["box"], old["box"])
            if iou >= threshold:
                clash = (old, iou)
                break
        if clash is None:
            kept.append(item)
        elif log_list is not None:
            log_list.append({"stage": stage, "reason": f"iou={clash[1]:.2f}>={threshold}",
                             "dropped": brief(item), "kept": brief(clash[0])})
    return kept


# ---------- v3: geometry-aware dedup ----------
def pair_metrics(a, b):
    """(iou, khoang cach tam chuan hoa theo duong cheo nho nhat, ti le dien tich nho/lon)."""
    iou = calculate_iou(a, b)
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    dist = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2.0, (a[1] + a[3] - b[1] - b[3]) / 2.0)
    diag = max(min(math.hypot(aw, ah), math.hypot(bw, bh)), 1e-6)
    size = min(aw * ah, bw * bh) / max(aw * ah, bw * bh, 1e-6)
    return iou, dist / diag, size


def looks_like_two_objects(frame, a, b):
    """Vung hop cua 2 box co hinh chu X / nhieu vat khong? (minAreaRect fill thap).
    CHI dung lam 'phu quyet dedup' cho cap box gan nhu trung nhau, nen sai so nghieng ve giu lai
    (va phien dem bi gan co review 'kept_overlap_pair')."""
    if frame is None or np is None:
        return False
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(min(a[0], b[0]))), max(0, int(min(a[1], b[1])))
    x2, y2 = min(w, int(max(a[2], b[2]))), min(h, int(max(a[3], b[3])))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < 12:
        return False
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.mean() / 255.0 > 0.5:            # vat the la phan nho hon cua ROI
        m = 255 - m
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    fill = cv2.contourArea(c) / max(rw * rh, 1.0)
    return fill < FILL_X_MAX              # 1 vit don: fill cao; chu X: fill thap


def geo_dedup(dets, frame=None, log_list=None, stage="geo_dedup"):
    """Chi loai box THUC SU trung (gan nhu giong het). Hai box chi giao nhau -> giu ca hai."""
    kept = []
    for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
        drop = None
        for k in kept:
            iou, cd, size = pair_metrics(d["box"], k["box"])
            if d["class_id"] == k["class_id"]:
                dup = iou >= DUP_IOU_SAME and cd <= DUP_CDIST and size >= DUP_SIZE
                if dup and looks_like_two_objects(frame, d["box"], k["box"]):
                    dup = False
                    d["kept_overlap_pair"] = True
                    k["kept_overlap_pair"] = True
            else:
                dup = iou >= DUP_IOU_CROSS          # cung 1 vat bi gan 2 class
            if dup:
                drop = (k, iou)
                break
        if drop is None:
            kept.append(d)
        elif log_list is not None:
            log_list.append({"stage": stage, "reason": f"near_identical iou={drop[1]:.2f}",
                             "dropped": brief(d), "kept": brief(drop[0])})
    return kept


def predict_frame(frame, cfg, dropped, frame_index=0):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_roi_px(w, h)
    crop = frame[y1:y2, x1:x2]
    source = preprocess_frame(crop) if cfg["preprocess"] else crop
    scales = cfg.get("scales")
    imgsz = scales[frame_index % len(scales)] if scales else cfg["imgsz"]
    kw = dict(source=source, conf=cfg["conf_infer"], imgsz=imgsz,
              agnostic_nms=cfg["agnostic"], iou=cfg["nms_iou"], max_det=300, verbose=False)
    if cfg.get("nms_free"):
        kw.update(E2E_KW)
    result = model.predict(**kw)[0]
    raw = get_detections(result)
    for det in raw:  # crop -> frame goc
        bx1, by1, bx2, by2 = det["box"]
        det["box"] = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
    local_drop = []
    if cfg.get("geo"):
        kept = geo_dedup(raw, frame, local_drop, f"frame{frame_index + 1}_geo")
    else:
        kept = deduplicate(raw, log_list=local_drop, stage=f"frame{frame_index + 1}_dedup")
    dropped.extend(local_drop)
    info = {"raw": len(raw), "kept": len(kept), "imgsz": imgsz,
            "low_conf": sum(1 for d in kept if d["confidence"] < CONFIDENCE),
            "dets": [brief(d) for d in kept] if DEBUG else None}
    return kept, info


def shape_evidence(frame, box):
    """Bang chung hinh dang nut/washer; None neu phan doan khong dang tin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pad = max(3, int(max(x2 - x1, y2 - y1) * 0.18))
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or min(roi.shape[:2]) < SHAPE_MIN_SIZE:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    candidates = []
    for polarity in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            roi_area = roi.shape[0] * roi.shape[1]
            if not 0.08 * roi_area < area < 0.78 * roi_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bx <= 1 or by <= 1 or bx + bw >= roi.shape[1] - 1 or by + bh >= roi.shape[0] - 1:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area else 0
            approx = cv2.approxPolyDP(hull, 0.035 * cv2.arcLength(hull, True), True)
            if not 0.78 <= solidity <= 1.02:
                continue
            candidates.append((area / roi_area, circularity, len(approx)))
    if not candidates:
        return None
    _, circularity, vertices = max(candidates)
    if 5 <= vertices <= 7 and 0.78 <= circularity <= 0.94:
        return "nut", circularity, vertices
    if vertices >= 8 and circularity >= 0.95:
        return "washer", circularity, vertices
    return None


def apply_shape_check(frame, detections, cfg):
    """Luon tinh bang chung hinh dang (de canh bao/review).
    Chi DOI class khi cfg['shape_correct'] bat (baseline) hoac SHAPE_CORRECT=1."""
    corrected = []
    class_lookup = {str(v).lower(): int(k) for k, v in model.names.items()}
    for index, item in enumerate(detections, start=1):
        d = dict(item)
        name = cname(d["class_id"])
        d["original_class"] = name
        d["shape"] = None
        if USE_SHAPE_CHECK and name in ("nut", "washer"):
            evidence = shape_evidence(frame, d["box"])
            if SHAPE_DEBUG:
                log(f"[SHAPE DEBUG] #{index} YOLO={name} conf={d['confidence']:.3f} "
                    f"box={[round(v) for v in d['box']]} OpenCV={evidence}")
            if evidence is not None:
                shape_class, circularity, vertices = evidence
                d["shape"] = {"class": shape_class, "circularity": round(circularity, 3),
                              "vertices": vertices}
                if shape_class != name:
                    strong = ((shape_class == "nut" and circularity <= 0.90 and vertices == 6)
                              or (shape_class == "washer" and circularity >= 0.97 and vertices >= 9))
                    if strong and cfg["shape_correct"] and shape_class in class_lookup:
                        d["class_id"] = class_lookup[shape_class]
                        d["shape_changed_class"] = True
                        log(f"[SHAPE CORRECTION] #{index}: {name} -> {shape_class}")
                    elif strong:
                        log(f"[SHAPE SUGGEST] #{index}: {name} -> {shape_class} (khong ap dung, chi canh bao)")
        corrected.append(d)
    return corrected


# ==================================================
# 3. GHEP 5 FRAME
# ==================================================
def merge_detections_legacy(frame_detections, dropped):
    """Giu nguyen logic cu (baseline). Luu y: nguong 'conf<0.60' o day khong con y nghia
    vi CONFIDENCE hien la 0.5 (moi detection da >= nguong infer)."""
    tracks = []
    for frame_index, detections in enumerate(frame_detections):
        used = set()
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            best_idx, best_iou = None, 0.0
            for i, track in enumerate(tracks):
                if i in used or frame_index - track["last_frame"] > 2:
                    continue
                iou = calculate_iou(det["box"], track["last_box"])
                if iou > best_iou:
                    best_idx, best_iou = i, iou
            if best_idx is not None and best_iou >= IOU_THRESHOLD:
                track = tracks[best_idx]
                track["observations"].append(det)
                track["last_box"], track["last_frame"] = det["box"], frame_index
                used.add(best_idx)
            else:
                tracks.append({"observations": [det], "last_box": det["box"], "last_frame": frame_index})
                used.add(len(tracks) - 1)
    merged = []
    for track in tracks:
        obs = track["observations"]
        if len(obs) < 2 and max(d["confidence"] for d in obs) < 0.60:
            dropped.append({"stage": "merge", "reason": "single_obs_low_conf", "dropped": brief(obs[0])})
            continue
        merged.append(_finalize_track(obs, "confirmed"))
    return deduplicate(merged, log_list=dropped, stage="post_merge_dedup")


def merge_detections_v2(frame_detections, cfg, dropped):
    """Matching toan cuc theo IoU (cap IoU cao nhat truoc). Track xac nhan neu:
      - max conf >= CONFIDENCE, HOAC
      - xuat hien >= RECOVER_MIN_OBS frame va conf TB >= RECOVER_MIN_MEAN (status=recovered)."""
    tracks = []
    for fi, dets in enumerate(frame_detections):
        pairs = []
        for di, d in enumerate(dets):
            for ti, t in enumerate(tracks):
                if t["last_frame"] == fi or fi - t["last_frame"] > 2:
                    continue
                iou = calculate_iou(d["box"], t["last_box"])
                if iou >= IOU_THRESHOLD:
                    pairs.append((iou, di, ti))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for iou, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            tracks[ti]["observations"].append(dets[di])
            tracks[ti]["last_box"], tracks[ti]["last_frame"] = dets[di]["box"], fi
            used_d.add(di)
            used_t.add(ti)
        for di, d in enumerate(dets):
            if di not in used_d:
                tracks.append({"observations": [d], "last_box": d["box"], "last_frame": fi})
    merged = []
    for t in tracks:
        obs = t["observations"]
        confs = [d["confidence"] for d in obs]
        mx, mean = max(confs), sum(confs) / len(confs)
        if mx >= CONFIDENCE:
            merged.append(_finalize_track(obs, "confirmed"))
        elif cfg["recover"] and len(obs) >= RECOVER_MIN_OBS and mean >= RECOVER_MIN_MEAN:
            merged.append(_finalize_track(obs, "recovered"))
        else:
            dropped.append({"stage": "merge",
                            "reason": f"unconfirmed(n={len(obs)},max={mx:.2f},mean={mean:.2f})",
                            "dropped": brief(max(obs, key=lambda d: d["confidence"]))})
    return deduplicate(merged, log_list=dropped, stage="post_merge_dedup")


# ---------- v3: Hungarian + chi phi hinh hoc ----------
def match_cost(det, ref):
    """Chi phi ghep 1 detection voi 1 cum (box tham chieu = trung binh). None = khong the ghep.
    Class khac chi bi PHAT mem (khong chan cung)."""
    iou, cd, size = pair_metrics(det["box"], ref["box"])
    if iou < 0.15 and cd > 0.35:
        return None
    cls_pen = 0.0 if det["class_id"] == ref["class_id"] else 0.4
    return (1 - iou) + 0.5 * cd + 0.3 * (1 - size) + cls_pen


def _assign(C):
    """Tra ve list (hang, cot). Dung scipy neu co, neu khong thi greedy theo chi phi tang dan."""
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(C)
        return list(zip(r.tolist(), c.tolist()))
    cells = sorted((C[i, j], i, j) for i in range(C.shape[0]) for j in range(C.shape[1]))
    used_r, used_c, out = set(), set(), []
    for _, i, j in cells:
        if i not in used_r and j not in used_c:
            used_r.add(i)
            used_c.add(j)
            out.append((i, j))
    return out


def merge_detections_v3(frame_detections, cfg, dropped, frame=None):
    """Vat dung yen -> cum theo vi tri. Moi frame ghep MOT-MOT (Hungarian) nen hai vit cung frame
    khong bao gio bi gop thanh mot. Doi cho 2 box gan trung nhau khong lam sai so dem."""
    clusters = []
    for fi, dets in enumerate(frame_detections):
        matched = {}
        if clusters and dets and np is not None:
            C = np.full((len(dets), len(clusters)), _BIG)
            for i, d in enumerate(dets):
                for j, c in enumerate(clusters):
                    cost = match_cost(d, c)
                    if cost is not None and cost <= MATCH_MAX_COST:
                        C[i, j] = cost
            for i, j in _assign(C):
                if C[i, j] < _BIG:
                    matched[i] = j
        for i, d in enumerate(dets):
            if i in matched:
                c = clusters[matched[i]]
                c["obs"].append(d)
                n = len(c["obs"])
                c["box"] = [sum(o["box"][k] for o in c["obs"]) / n for k in range(4)]
                c["class_id"] = max(c["obs"], key=lambda o: o["confidence"])["class_id"]
            else:
                clusters.append({"obs": [d], "box": list(d["box"]), "class_id": d["class_id"]})
    merged = []
    for c in clusters:
        obs = c["obs"]
        confs = [o["confidence"] for o in obs]
        mx, mean = max(confs), sum(confs) / len(confs)
        if mx >= CONFIDENCE:
            merged.append(_finalize_track(obs, "confirmed"))
        elif cfg["recover"] and len(obs) >= RECOVER_MIN_OBS and mean >= RECOVER_MIN_MEAN:
            merged.append(_finalize_track(obs, "recovered"))
        else:
            dropped.append({"stage": "merge",
                            "reason": f"unconfirmed(n={len(obs)},max={mx:.2f},mean={mean:.2f})",
                            "dropped": brief(max(obs, key=lambda o: o["confidence"]))})
    return geo_dedup(merged, frame, dropped, "post_merge_geo")


def _finalize_track(obs, status):
    votes = defaultdict(float)
    for d in obs:
        votes[d["class_id"]] += d["confidence"]
    chosen = max(votes, key=votes.get)
    selected = [d for d in obs if d["class_id"] == chosen]
    return {
        "box": [sum(d["box"][j] for d in selected) / len(selected) for j in range(4)],
        "class_id": chosen,
        "confidence": sum(d["confidence"] for d in selected) / len(selected),
        "observations": len(obs),
        "status": status,
        "class_share": votes[chosen] / sum(votes.values()),
        "votes": {cname(k): round(v, 3) for k, v in votes.items()},
        "kept_overlap_pair": any(d.get("kept_overlap_pair") for d in obs),
    }


# ---------- v3: re-inference vung nghi ngo ----------
def refine_suspects(frame, merged, cfg, dropped):
    """Box screw to bat thuong (nghi 2 vit bi gop) -> chay lai YOLO tren vung crop phong lon.
    Neu detector tu tach duoc >=2 screw conf>=CONFIDENCE thi thay the, nhung danh dau
    'recovered' (CHO XAC MINH) - khong tu tang so 'da xac nhan'. Toi da MAX_REFINE vung/phien."""
    if not cfg.get("refine"):
        return merged
    screw_areas = sorted((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
                         for d in merged if cname(d["class_id"]) == "screw")
    if len(screw_areas) < 3:
        return merged
    med = screw_areas[len(screw_areas) // 2]
    H, W = frame.shape[:2]
    out, n_ref = [], 0
    for d in merged:
        x1, y1, x2, y2 = d["box"]
        area = (x2 - x1) * (y2 - y1)
        suspect = cname(d["class_id"]) == "screw" and area > SUSPECT_AREA_K * med
        if not suspect or n_ref >= MAX_REFINE:
            out.append(d)
            continue
        n_ref += 1
        pad = 0.15 * max(x2 - x1, y2 - y1)
        cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        cx2, cy2 = min(W, int(x2 + pad)), min(H, int(y2 + pad))
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            out.append(d)
            continue
        r = model.predict(frame[cy1:cy2, cx1:cx2], conf=cfg["conf_infer"], imgsz=REFINE_IMGSZ,
                          max_det=50, verbose=False, **E2E_KW)[0]
        subs = get_detections(r)
        for s in subs:
            s["box"] = [s["box"][0] + cx1, s["box"][1] + cy1, s["box"][2] + cx1, s["box"][3] + cy1]
        subs = geo_dedup([s for s in subs if cname(s["class_id"]) == "screw"], frame)
        # chi giu cac box co tam nam trong box nghi ngo goc
        subs = [s for s in subs
                if x1 <= (s["box"][0] + s["box"][2]) / 2 <= x2 and y1 <= (s["box"][1] + s["box"][3]) / 2 <= y2]
        if len(subs) >= 2 and all(s["confidence"] >= CONFIDENCE for s in subs):
            for s in subs:
                s.update(status="recovered", observations=1, class_share=1.0,
                         votes={"screw": round(s["confidence"], 3)}, from_refine=True)
                out.append(s)
            dropped.append({"stage": "refine", "reason": f"split_into_{len(subs)}", "dropped": brief(d)})
            log(f"[REFINE] box {brief(d)['box']} tach thanh {len(subs)} screw (cho xac minh)")
        else:
            out.append(d)
    return out


def review_reasons_for_session(frame_detections, merged_before_shape, merged_after_shape):
    reasons = []
    frame_counts = [len(dets) for dets in frame_detections]
    if frame_counts and max(frame_counts) - min(frame_counts) >= REVIEW_COUNT_SPREAD:
        reasons.append(f"frame_count_variation:{frame_counts}")
    for idx, det in enumerate(merged_after_shape, 1):
        name = cname(det["class_id"])
        if name in ("nut", "washer") and det["confidence"] < REVIEW_CONFIDENCE:
            reasons.append(f"low_confidence:object_{idx}:{name}:{det['confidence']:.3f}")
        if det.get("observations", 0) < 3:
            reasons.append(f"few_observations:object_{idx}:{det.get('observations', 0)}")
        if det.get("status") == "recovered":
            reasons.append(f"recovered_low_conf:object_{idx}:{name}:{det['confidence']:.3f}")
        if det.get("class_share", 1.0) < CLASS_SHARE_REVIEW:
            reasons.append(f"class_unstable:object_{idx}:{det['votes']}")
        if det.get("kept_overlap_pair"):
            reasons.append(f"kept_overlap_pair:object_{idx}:{name}")
        if det.get("from_refine"):
            reasons.append(f"split_suspect:object_{idx}:{name}")
        if REVIEW_ON_SHAPE_DISAGREEMENT and det.get("shape"):
            if det["shape"]["class"] != det.get("original_class"):
                reasons.append(f"shape_disagreement:object_{idx}")
    for idx, det in enumerate(merged_before_shape, 1):
        votes = Counter()
        for frame_dets in frame_detections:
            matches = [(calculate_iou(det["box"], x["box"]), x) for x in frame_dets]
            if matches:
                iou, match = max(matches, key=lambda p: p[0])
                if iou >= IOU_THRESHOLD:
                    votes[cname(match["class_id"])] += 1
        if votes.get("nut", 0) and votes.get("washer", 0):
            reasons.append(f"nut_washer_vote_conflict:object_{idx}:{dict(votes)}")
    return reasons


def run_counting_pipeline(frames, cfg=None):
    """Pipeline thuan (khong dung camera/Flask) -> dung chung cho app va cac script offline."""
    cfg = cfg or make_cfg()
    dropped, per_frame, frame_dets = [], [], []
    t0 = time.perf_counter()
    for i, frame in enumerate(frames):
        dets, info = predict_frame(frame, cfg, dropped, frame_index=i)
        frame_dets.append(dets)
        per_frame.append(info)
        log(f"[FRAME {i + 1}/{len(frames)}] imgsz={info['imgsz']} raw={info['raw']} "
            f"after_dedup={info['kept']} below_{CONFIDENCE}={info['low_conf']}")
    t_infer = time.perf_counter() - t0

    best_frame, best_index, scores = select_best_frame(frames)   # can truoc khi ghep (geo-dedup dung anh)

    if cfg["legacy_merge"]:
        merged = merge_detections_legacy(frame_dets, dropped)
    elif cfg.get("hungarian"):
        merged = merge_detections_v3(frame_dets, cfg, dropped, best_frame)
    else:
        merged = merge_detections_v2(frame_dets, cfg, dropped)
    merged = refine_suspects(best_frame, merged, cfg, dropped)

    n_rec = sum(1 for d in merged if d.get("status") == "recovered")
    log(f"[MERGE] profile={cfg['name']} tracks_final={len(merged)} recovered={n_rec} "
        f"dropped={len(dropped)} infer={t_infer:.2f}s")
    if DEBUG:
        for item in dropped:
            log(f"[DROP] {item['stage']} {item['reason']} {item['dropped']}")

    baseline = [dict(d) for d in merged]
    final = apply_shape_check(best_frame, merged, cfg)
    reasons = review_reasons_for_session(frame_dets, baseline, final)
    # tong phat hien = da xac nhan (tu dong) + cho xac minh (conf thap / tach tu refine, can nguoi kiem tra).
    counts = Counter(cname(d["class_id"]) for d in final)
    counts_confirmed = Counter(cname(d["class_id"]) for d in final if d.get("status") != "recovered")
    counts_pending = Counter(cname(d["class_id"]) for d in final if d.get("status") == "recovered")
    n_pending = sum(counts_pending.values())
    log(f"[COMPARE] YOLO+merge: {dict(Counter(cname(d['class_id']) for d in baseline))}")
    log(f"[COUNT] total={len(final)} confirmed={len(final) - n_pending} pending={n_pending} "
        f"by_class={dict(counts)} pending_by_class={dict(counts_pending)}")
    return {"cfg": cfg, "merged": final, "baseline": baseline, "best_frame": best_frame,
            "best_index": best_index, "sharpness": scores, "frame_detections": frame_dets,
            "per_frame": per_frame, "dropped": dropped, "reasons": reasons,
            "counts": counts, "counts_confirmed": counts_confirmed, "counts_pending": counts_pending,
            "total": len(final), "confirmed": len(final) - n_pending, "pending": n_pending,
            "t_infer": t_infer, "t_total": time.perf_counter() - t0}


# ---------- chu tieng Viet tren anh ket qua ----------
_BOLD_FONTS = [
    r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\tahomabd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
_font_cache = {}


def _load_font(px):
    """Tim font dam ho tro tieng Viet. Co the chi dinh bang bien moi truong OVERLAY_FONT."""
    if px in _font_cache or ImageFont is None:
        return _font_cache.get(px)
    font = None
    env = os.getenv("OVERLAY_FONT")
    for path in ([env] if env else []) + _BOLD_FONTS:
        try:
            font = ImageFont.truetype(path, px)
            break
        except Exception:
            continue
    _font_cache[px] = font
    return font


def strip_accents(text):
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def draw_summary_panel(img, lines, k):
    """Ve bang tong ket goc tren-trai (nen den mo + chu mau). lines = [(text, bgr), ...].
    Dung font TrueType nen hien thi duoc dau tieng Viet; neu khong tim thay font thi ve chu khong dau."""
    h, w = img.shape[:2]
    size, pad, step = int(round(28 * k)), int(round(14 * k)), int(round(40 * k))
    x0, y0 = int(15 * k), int(10 * k)
    font = _load_font(size)
    if font is None:                                    # du phong: chu khong dau bang cv2
        fb = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.75 * k, max(2, int(round(2 * k)))
        txt = [(strip_accents(t), c) for t, c in lines]
        width = max(cv2.getTextSize(t, fb, fs, th)[0][0] for t, _ in txt) + 2 * pad
        x1, y1 = min(w, x0 + width), min(h, y0 + step * len(txt) + pad)
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8) if np is not None else img[y0:y1, x0:x1]
        for i, (t, c) in enumerate(txt):
            cv2.putText(img, t, (x0 + pad, y0 + int(round(30 * k)) + i * step), fb, fs, c, th, cv2.LINE_AA)
        return
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(max(probe.textlength(t, font=font) for t, _ in lines)) + 2 * pad
    x1, y1 = min(w, x0 + width), min(h, y0 + step * len(lines) + pad)
    roi = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)   # nen den, do dac 65%
    pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for i, (text, bgr) in enumerate(lines):
        draw.text((pad, pad // 2 + i * step), text, font=font, fill=(bgr[2], bgr[1], bgr[0]))
    img[y0:y1, x0:x1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


PENDING_BGR = (0, 140, 255)   # cam - trung voi the "Cho xac minh" tren giao dien
CLASS_BGR = {"nut": (255, 255, 0), "screw": (255, 160, 80), "washer": (255, 255, 255)}


def pending_items(merged):
    """Danh sach vat cho xac minh, danh so ?1, ?2... theo thu tu trong merged
    (cung thu tu duoc ve tren anh ket qua)."""
    items = []
    for d in merged:
        if d.get("status") == "recovered":
            items.append({"index": len(items) + 1, "class": cname(d["class_id"]),
                          "conf": round(d["confidence"], 3), "obs": d.get("observations"),
                          "box": [int(round(v)) for v in d["box"]]})
    return items


def annotate_result(frame, merged, res):
    """Xanh la = da xac nhan; CAM + nhan '?n' = cho xac minh (conf thap / tach tu refine)."""
    annotated = frame.copy()
    k = max(1.0, frame.shape[1] / 1920.0)       # anh 4K -> chu va net ve to hon cho de doc
    font = cv2.FONT_HERSHEY_SIMPLEX
    pend_no = 0
    for det in merged:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        pend = det.get("status") == "recovered"
        name = str(model.names[det["class_id"]])
        if pend:
            pend_no += 1
            label, color = f"?{pend_no} {name} {det['confidence']:.2f}", PENDING_BGR
        else:
            label, color = f"{name}: {det['confidence']:.2f}", (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, int(round((3 if pend else 2) * k)))
        cv2.putText(annotated, label, (x1, max(int(20 * k), y1 - int(8 * k))),
                    font, 0.55 * k, color, max(2, int(round(2 * k))))

    counts, pend_c = res["counts"], res["counts_pending"]

    def cls_line(c, label):
        extra = f"  (chờ {pend_c[c]})" if pend_c.get(c) else ""
        return (f"{label}: {counts.get(c, 0)}{extra}", CLASS_BGR[c])

    lines = [(f"Tổng phát hiện: {res['total']}", (255, 255, 255)),
             (f"Đã xác nhận: {res['confirmed']}", (0, 255, 0)),
             (f"Chờ xác minh: {res['pending']}", PENDING_BGR),
             cls_line("nut", "Đai ốc"), cls_line("screw", "Ốc vít"), cls_line("washer", "Long đền")]
    draw_summary_panel(annotated, lines, k)
    return annotated, lines


def build_report(res, filename, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = get_roi_px(w, h)
    cfg = res["cfg"]
    report = {
        "result_filename": filename, "time": datetime.now().isoformat(timespec="seconds"),
        "profile": cfg["name"], "params": {k: cfg[k] for k in cfg},
        "e2e_kwargs": E2E_KW,
        "confidence_final": CONFIDENCE, "dedup_iou": DEDUP_IOU, "merge_iou": IOU_THRESHOLD,
        "geo_dedup": {"iou_same": DUP_IOU_SAME, "cdist": DUP_CDIST, "size": DUP_SIZE,
                      "iou_cross": DUP_IOU_CROSS, "fill_x_max": FILL_X_MAX},
        "roi_px": [x1, y1, x2, y2], "frame_size": [w, h],
        "counts": dict(res["counts"]), "counts_confirmed": dict(res["counts_confirmed"]),
        "counts_pending": dict(res["counts_pending"]),
        "total": res["total"], "confirmed": res["confirmed"], "pending": res["pending"],
        "pending_items": pending_items(res["merged"]),
        "sharpness": [round(s, 2) for s in res["sharpness"]], "best_frame": res["best_index"] + 1,
        "timing_s": {"infer": round(res["t_infer"], 3), "total": round(res["t_total"], 3)},
        "review_reasons": res["reasons"],
        "objects": [{**brief(d), "status": d.get("status"), "obs": d.get("observations"),
                     "class_share": round(d.get("class_share", 1.0), 3), "votes": d.get("votes"),
                     "original_class": d.get("original_class"), "shape": d.get("shape"),
                     "shape_changed_class": d.get("shape_changed_class", False),
                     "kept_overlap_pair": d.get("kept_overlap_pair", False),
                     "from_refine": d.get("from_refine", False)}
                    for d in res["merged"]],
        "per_frame": [{k: v for k, v in i.items() if k != "dets"} for i in res["per_frame"]],
        "dropped_count": len(res["dropped"]),
    }
    if DEBUG:
        report["debug"] = {"dropped": res["dropped"],
                           "per_frame_dets": [i["dets"] for i in res["per_frame"]]}
    return report


def process_count_session(frames, current_session_id):
    global session_state, session_message
    global latest_counts, latest_total, latest_detections, latest_recovered
    global latest_result_jpeg, latest_result_filename
    global latest_review_frame, latest_review_metadata, latest_review_reasons
    global latest_report
    try:
        if not frames:
            raise RuntimeError("Khong thu thap duoc frame.")
        res = run_counting_pipeline(frames, make_cfg())
        merged, best_frame, counts, total = res["merged"], res["best_frame"], res["counts"], res["total"]
        annotated, lines = annotate_result(best_frame, merged, res)

        log(f"\n========== KET QUA DEM ({len(frames)} FRAME, profile={res['cfg']['name']}) ==========")
        for line, _ in lines:
            log(line)
        log("============================================\n")

        # Khong ghi anh ra dia: giu JPEG trong RAM, chi phuc vu hien thi qua /latest_result.
        filename = f"result_{datetime.now():%Y%m%d_%H%M%S}"
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            raise RuntimeError("Khong ma hoa duoc anh ket qua.")

        report = build_report(res, filename, best_frame.shape)

        reasons = res["reasons"]
        if reasons:
            log(f"[REVIEW] {len(reasons)} flags: {reasons}")

        n_rec = res["pending"]
        with lock:
            if current_session_id != session_id or not camera_enabled:
                return
            latest_review_frame = best_frame.copy()
            latest_review_metadata = {"result_filename": filename}
            latest_review_reasons = reasons
            latest_counts = dict(counts)
            latest_total = total
            latest_detections = res["total"]
            latest_recovered = n_rec
            latest_report = report
            latest_result_jpeg = encoded.tobytes()
            latest_result_filename = filename
            session_state = "done"
            session_message = (f"Hoan tat: {total} phat hien ({res['confirmed']} da xac nhan, "
                               f"{n_rec} cho xac minh).")
    except Exception as exc:
        log(f"[ERROR] process_count_session: {exc!r}")
        with lock:
            if current_session_id == session_id:
                session_state = "error"
                session_message = f"Loi nhan dien: {exc}"


# ==================================================
# 5. CAMERA RTSP
# ==================================================
def open_ip_camera():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _abort_session_locked(message):
    """Goi khi dang giu lock: huy phien dang countdown/sampling neu camera rot."""
    global session_state, session_message, sample_frames
    if session_state in ("countdown", "sampling"):
        session_state = "error"
        session_message = message
        sample_frames = []


def camera_worker():
    global camera_enabled, camera_ready, camera_status
    global latest_frame, latest_jpeg
    global session_state, session_message
    global sample_frames, last_sample_time

    cap = None
    ready_reported = False
    last_preview = 0.0
    try:
        while True:
            with lock:
                if not camera_enabled:
                    break

            if cap is None or not cap.isOpened():
                ready_reported = False   # reset de camera_ready bat lai duoc sau khi rot mang
                with lock:
                    camera_ready = False
                    camera_status = "Đang kết nối camera IP..."
                    session_message = "Đang kết nối Hikvision RTSP..."
                    _abort_session_locked("Mất kết nối camera trong lúc đếm - hãy đếm lại")
                if cap is not None:
                    cap.release()
                cap = open_ip_camera()
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    with lock:
                        camera_status = "Mất kết nối camera IP - đang thử lại"
                    time.sleep(RTSP_RECONNECT_DELAY)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                ready_reported = False
                with lock:
                    camera_ready = False
                    camera_status = "Mất tín hiệu RTSP - đang kết nối lại"
                    _abort_session_locked("Mất tín hiệu camera trong lúc đếm - hãy đếm lại")
                time.sleep(RTSP_RECONNECT_DELAY)
                continue

            if not ready_reported:
                ready_reported = True
                with lock:
                    camera_ready = True
                    camera_status = "Camera IP đang bật"
                    session_message = "Camera IP đã sẵn sàng"

            now = time.monotonic()
            frames_to_process = None
            process_session_id = None
            with lock:
                latest_frame = frame  # khong copy: frame moi moi vong lap, cac noi dung can thi tu copy
                if session_state == "countdown" and now >= countdown_deadline:
                    session_state = "sampling"
                    session_message = f"Đang lấy {SAMPLE_COUNT} frame..."
                    sample_frames = []
                    last_sample_time = 0.0
                if session_state == "sampling":
                    if not sample_frames or now - last_sample_time >= SAMPLE_INTERVAL:
                        sample_frames.append(frame.copy())
                        last_sample_time = now
                    if len(sample_frames) >= SAMPLE_COUNT:
                        frames_to_process = [item.copy() for item in sample_frames]
                        process_session_id = session_id
                        session_state = "processing"
                        session_message = f"Đang chạy YOLO trên {SAMPLE_COUNT} frame và ghép kết quả..."

            # Preview: gioi han PREVIEW_FPS de giam tai CPU (imencode 1080p moi frame rat ton)
            if now - last_preview >= 1.0 / PREVIEW_FPS:
                last_preview = now
                preview = frame.copy()
                if USE_ROI:
                    ph, pw = preview.shape[:2]
                    rx1, ry1, rx2, ry2 = get_roi_px(pw, ph)
                    cv2.rectangle(preview, (rx1, ry1), (rx2 - 1, ry2 - 1), (0, 255, 0), 2, cv2.LINE_AA)
                    label = f"ROI - YOLO AREA ({rx2 - rx1}x{ry2 - ry1}) [{PROFILE}]"
                    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
                    ly = max(th + 10, ry1 - 8)
                    cv2.rectangle(preview, (rx1, ly - th - 8), (rx1 + tw + 10, ly + base - 4), (0, 160, 0), -1)
                    cv2.putText(preview, label, (rx1 + 5, ly - 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with lock:
                        latest_jpeg = encoded.tobytes()

            if frames_to_process is not None:
                Thread(target=process_count_session,
                       args=(frames_to_process, process_session_id), daemon=True).start()
            time.sleep(0.005)
    finally:
        if cap is not None:
            cap.release()
        with lock:
            camera_enabled = False
            camera_ready = False
            camera_status = "Đã dừng"
            session_state = "idle"
            session_message = "Camera IP đã dừng"


# ==================================================
# 6. FLASK ROUTES
# ==================================================
@app.route("/pick")
def pick():
    """Giao dien chinh nam o /pick."""
    return render_template("index.html")


@app.route("/")
def index():
    # Giu "/" de ban cu/duong dan cu van vao duoc: chuyen sang /pick.
    return redirect("/pick")


@app.route("/start", methods=["POST"])
def start_camera():
    global camera_enabled, camera_thread
    if not CAMERA_PASSWORD:
        return jsonify(ok=False, message="Chưa đặt biến môi trường CAMERA_PASSWORD"), 400
    with lock:
        if camera_enabled:
            return jsonify(ok=True, message="Camera đang bật")
        camera_enabled = True
        camera_thread = Thread(target=camera_worker, daemon=True)
        camera_thread.start()
    return jsonify(ok=True, message="Đang khởi động camera...")


@app.route("/stop", methods=["POST"])
def stop_camera():
    global camera_enabled, session_id, session_state, session_message
    with lock:
        camera_enabled = False
        session_id += 1
        session_state = "idle"
        session_message = "Đã dừng camera"
    return jsonify(ok=True, message="Đã yêu cầu dừng camera")


@app.route("/start_count", methods=["POST"])
def start_count():
    global session_id, session_state, session_message
    global countdown_deadline, sample_frames
    global latest_counts, latest_total, latest_detections, latest_recovered
    global latest_result_jpeg, latest_result_filename
    global latest_review_frame, latest_review_metadata, latest_review_reasons
    global latest_report
    with lock:
        if not camera_enabled or not camera_ready:
            return jsonify(ok=False, message="Hãy bật camera và đợi camera sẵn sàng."), 400
        if session_state in ("countdown", "sampling", "processing"):
            return jsonify(ok=False, message="Phiên đếm hiện tại chưa kết thúc."), 400
        session_id += 1
        session_state = "countdown"
        session_message = "Chuẩn bị đếm ngược 3 giây..."
        countdown_deadline = time.monotonic() + COUNTDOWN_SECONDS
        sample_frames = []
        latest_counts = {}
        latest_total = 0
        latest_detections = 0
        latest_recovered = 0
        latest_result_jpeg = None
        latest_result_filename = None
        latest_review_frame = None
        latest_review_metadata = None
        latest_review_reasons = []
        latest_report = None
    return jsonify(ok=True, message="Bắt đầu đếm ngược 3 giây")


def generate_frames():
    while True:
        with lock:
            frame = latest_jpeg
            active = camera_enabled
        if frame is not None and active:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1.0 / PREVIEW_FPS)   # tranh vong lap quay 100% CPU khi frame khong doi
        else:
            time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/counts")
def get_counts():
    with lock:
        countdown_remaining = 0
        if session_state == "countdown":
            countdown_remaining = max(0, math.ceil(countdown_deadline - time.monotonic()))
        rep = latest_report or {}
        return jsonify(
            counts=latest_counts, total=latest_total, detections=latest_detections,
            confirmed=rep.get("confirmed", 0), pending=latest_recovered,
            counts_confirmed=rep.get("counts_confirmed", {}),
            counts_pending=rep.get("counts_pending", {}),
            pending_items=rep.get("pending_items", []),
            recovered=latest_recovered, profile=PROFILE,
            camera_status=camera_status, camera_ready=camera_ready,
            session_state=session_state, session_message=session_message,
            countdown_remaining=countdown_remaining, result_filename=latest_result_filename,
            review_required=bool(latest_review_reasons), review_reasons=latest_review_reasons,
        )


@app.route("/last_report")
def last_report():
    """Bao cao JSON phien gan nhat (ROI, tham so, tung vat the, vote, ly do loai)."""
    with lock:
        report = latest_report
    if report is None:
        return jsonify(ok=False, message="Chưa có báo cáo"), 404
    return jsonify(report)


@app.route("/latest_result")
def latest_result():
    with lock:
        data = latest_result_jpeg
    if data is None:
        return jsonify(ok=False, message="Chưa có ảnh kết quả"), 404
    return Response(data, mimetype="image/jpeg")


@app.route("/latest_raw")
def latest_raw():
    with lock:
        frame = latest_review_frame.copy() if latest_review_frame is not None else None
    if frame is None:
        return jsonify(ok=False, message="Chua co RAW"), 404
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return jsonify(ok=False, message="Khong ma hoa duoc RAW"), 500
    return Response(encoded.tobytes(), mimetype="image/png")


if __name__ == "__main__":
    cfg0 = make_cfg()
    log(f"[BOOT] profile={PROFILE} debug={DEBUG} imgsz={IMAGE_SIZE} scales={cfg0['scales']} "
        f"preprocess={USE_PREPROCESS} conf_infer={cfg0['conf_infer']} "
        f"shape_correct={cfg0['shape_correct']} geo={cfg0['geo']} refine={cfg0['refine']} "
        f"e2e_kwargs={E2E_KW}")
    if cfg0["nms_free"] and not E2E_KW:
        log("[BOOT][WARN] Khong nhan dien duoc tham so NMS-free (nms=False / end2end=True) - "
            "v3 se chay voi che do mac dinh cua model.")
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, threaded=True)