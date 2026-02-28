import os
import tempfile
import subprocess
import base64
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models
from facenet_pytorch import MTCNN
from lime import lime_image
from skimage.segmentation import slic
from scipy.ndimage import gaussian_filter

# ========== SETTINGS ==========
IMG_SIZE    = 224
MAX_SIDE    = 600
NUM_FRAMES  = 50
TOP_K       = 2
MAX_MB      = 10
MAX_SECS    = 60

# ========== PAGE CONFIG & THEME ==========
st.set_page_config(page_title="DeepFake Detection System", layout="wide", initial_sidebar_state="collapsed")
fake_thresh = st.sidebar.slider("Fake threshold (%)", 0, 100, 50)

# Wallpaper background
wall = "deep.avif"
with open(wall, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.markdown(f"""
<style>
  .stApp {{
    background: linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.78)),
                url('data:image/jpeg;base64,{b64}') no-repeat center/cover !important;
  }}
  /* Make the header transparent */
  header[data-testid="stHeader"] {{
    background-color: transparent !important;
  }}
</style>
""", unsafe_allow_html=True)

# ======== MODEL DEFINITION ==========
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
    def forward(self, x, states):
        h, c = states
        gates = self.i2h(x) + self.h2h(h)
        i, f, g, o = gates.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class ResNeXtLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        modules = list(backbone.children())[:-2]
        self.cnn = nn.Sequential(*modules, nn.AdaptiveAvgPool2d((1,1)))
        for p in self.cnn.parameters(): p.requires_grad = False
        self.rnn_cell = CustomLSTMCell(2048, 2048)
        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm1d(2048)
        self.fc   = nn.Linear(2048, 2)
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        B, T, C, H, W = x.shape
        xf = x.view(B * T, C, H, W)
        with torch.no_grad(): feats = self.cnn(xf).view(B, T, 2048)
        h = torch.zeros(B, 2048, device=feats.device)
        c = torch.zeros(B, 2048, device=feats.device)
        for t in range(T): h, c = self.rnn_cell(feats[:, t, :], (h, c))
        h = self.bn(self.relu(h))
        return self.drop(self.fc(h))

# ======== LOAD MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNeXtLSTM().to(device)
state = torch.load("Resnetlstm/Finalmodel/Dropout_rate/0.0/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(state, strict=True)
model.eval()

# ======== DETECTOR ==========
detector = MTCNN(image_size=IMG_SIZE, margin=0, keep_all=False, device=device)

# ======== HELPERS ==========
PARTS = ['left_eye','right_eye','nose','mouth_left','mouth_right']

def extract_frames(path):
    orig_imgs, tensors, lmks_list = [], [], []
    cap = cv2.VideoCapture(path)
    while len(orig_imgs) < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        aligned, lm = extract_face_and_landmarks(frame)
        if aligned is None: continue
        orig_imgs.append(aligned)
        tensors.append(torch.from_numpy(aligned).permute(2,0,1).float()/255.0)
        lmks_list.append(lm)
    cap.release()
    if not tensors:
        st.error("No faces detected.")
        st.stop()
    seq = torch.stack(tensors).unsqueeze(0)
    return orig_imgs, seq, lmks_list

def extract_face_and_landmarks(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_SIDE / max(h, w), 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = detector.detect(Image.fromarray(rgb), landmarks=True)
    if boxes is None or probs[0] < 0.7:
        return None, None
    x1, y1, x2, y2 = map(int, boxes[0])
    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        return None, None
    aligned = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    lm = np.array(landmarks[0])
    lm[:,0] = ((lm[:,0]-x1) * IMG_SIZE/(x2-x1)).astype(int)
    lm[:,1] = ((lm[:,1]-y1) * IMG_SIZE/(y2-y1)).astype(int)
    return aligned, lm

def batch_predict(images):
    # images: list of numpy HxWx3 arrays
    tensors = []
    for img in images:
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        else:
            # assume PIL Image
            t = transforms.ToTensor()(img)
        tensors.append(t)
    batch = torch.stack(tensors).unsqueeze(1).to(device)  # (N,1,3,H,W)
    with torch.no_grad():
        logits = model(batch)
    return torch.softmax(logits, dim=1).cpu().numpy()

status = st.empty()
progress = st.progress(0)

# ======== UI ==========
st.title("DeepFake Detection System")
video_file = st.file_uploader("Upload video", type=["mp4","mov","avi"], key="vid_upload", help="Limit 10 MB per file – MP4, MOV, AVI")
if not video_file:
    st.info("Please upload a video.")
    st.stop()



# Retrieve raw bytes once
data = video_file.read()
st.video(data)
# 1. Compress
status.text("1. Checking file size...")
size_mb = len(data)/(1024*1024)
if size_mb > MAX_MB:
    status.text("Compressing video...")
    tmpc = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmpc.write(data)
    path_in = tmpc.name
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    subprocess.run(["c23","compress", path_in, "-o", tmp2.name], check=True)
    path = tmp2.name
else:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data)
    path = tmp.name
progress.progress(0.25)
# 2. Check duration
status.text("2. Validating duration...")
cap_chk = cv2.VideoCapture(path)
secs = int(cap_chk.get(cv2.CAP_PROP_FRAME_COUNT) / cap_chk.get(cv2.CAP_PROP_FPS))
cap_chk.release()
if secs > MAX_SECS:
    st.error("Video too long. Must be <= 1 minute.")
    st.stop()
progress.progress(0.40)
# 3. Preprocess
status.text("3. Preprocessing frames...")
orig, seq, lmks_list = extract_frames(path)
progress.progress(0.60)
# 4. Prediction
status.text("4. Predicting...")
logits = model(seq.to(device))
probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
label = "Fake" if probs[1]*100>fake_thresh else "Real"
confidence = probs.max()*100
progress.progress(0.80)

# 5. Tabs
# ─── INSERT THIS ───────────────────────────────────────────────────────────────

# bump up the tab‐label font size & weight
st.markdown("""
<style>
  /* increase size & weight of tab labels */
  div[data-baseweb="tab-list"] [role="tab"] {
    font-size: 50rem !important;
    font-weight: 700 !important;
  }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────

# 5. Tabs
tab1, tab2 = st.tabs(["Predictions","Explanation"])
with tab1:
    st.markdown(f"### {label} ({confidence:.1f}% confidence)")
    
progress.progress(0.90)
with tab2:
    if label == "Fake":
        explainer = lime_image.LimeImageExplainer()
        scores = [torch.softmax(model(seq[:,i:i+1].to(device)),dim=1)[0,1].item() for i in range(NUM_FRAMES)]
        top_idxs = np.argsort(scores)[-TOP_K:][::-1]
        cols = st.columns(TOP_K)
        for j, idx in enumerate(top_idxs):
            frame_np = np.array(orig[idx])
            exp = explainer.explain_instance(
                image=frame_np,
                classifier_fn=batch_predict,
                top_labels=1,
                hide_color=0,
                num_samples=500,
                segmentation_fn=lambda x: slic(x, n_segments=50, compactness=1)
            )
            mask = exp.get_image_and_mask(
                label=exp.top_labels[0],
                positive_only=True,
                num_features=10,
                hide_rest=False
            )[1]
            lm = lmks_list[idx]
            heat = gaussian_filter(mask.astype(float), sigma=5)
            regions = {PARTS[i]: np.mean(heat[tuple(lm[i])]) for i in range(len(PARTS))}
            best = max(regions, key=regions.get)
            overlay = (frame_np/255.0 * 0.6 + (mask[...,None]>0)*np.array([1,0,0])*0.4)
            cols[j].image(orig[idx], caption=f"Frame {idx+1} Original", use_container_width=True)
            cols[j].image(overlay, caption=f"Mask on {best}", use_container_width=True)
            cols[j].markdown(f"**Explanation**: Evidence strongest around {best}.")
    else:
        st.success("Video appears Real. No artifacts detected.")
progress.progress(1.0)
status.text("Done.")
