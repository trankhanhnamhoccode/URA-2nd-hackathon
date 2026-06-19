# Hướng dẫn notebook `ura_hybrid_paddle_vietocr_resnet_gate.ipynb`

## 1. Notebook này làm gì?

Notebook này tạo file nộp cho bài URA theo pipeline:

```text
test image
  ↓
preprocess ảnh
  ↓
PaddleOCR GPU đọc OCR chính
  ↓
nếu PaddleOCR blank/yếu → VietOCR rescue đọc lại từ crop text
  ↓
post-process OCR text
  ↓
rule/fuzzy product predictor dự đoán product_name
  ↓
optional ResNet visual gate, mặc định tắt
  ↓
checkpoint_hybrid_paddle_vietocr_resnet.csv
  ↓
submission.csv
```

Ý tưởng chính: **PaddleOCR là engine chính**, còn **VietOCR chỉ cứu các case PaddleOCR yếu**, không thay toàn bộ pipeline. Phần `product_name` dùng rule/fuzzy và **không dùng LogisticRegression fallback** để tránh model tự bịa nhãn khi không chắc.

---

## 2. Luồng chạy theo từng block

### Block 0 — Giới thiệu notebook

Notebook ghi rõ đây là bản:

```text
Hybrid PaddleOCR + VietOCR Rescue + Optional ResNet Gate
```

Các quyết định chính:

- PaddleOCR GPU là OCR base.
- VietOCR chỉ chạy khi PaddleOCR blank/yếu.
- Product name dùng rule/fuzzy no-LR.
- ResNet gate là optional, mặc định tắt.
- Output cuối nằm ở:

```text
/kaggle/working/submission.csv
```

---

### Block 1–2 — Cài thư viện

Cell này cài lại môi trường OCR trên Kaggle:

```python
PADDLE_VERSION = "3.3.0"
PADDLE_CUDA = "cu118"
```

Nó làm các việc:

1. Gọi `nvidia-smi` để kiểm tra GPU.
2. Gỡ các package dễ xung đột:

```text
paddlepaddle
paddlepaddle-gpu
paddleocr
paddlex
torch
torchvision
torchaudio
vietocr
```

3. Cài `paddlepaddle-gpu`.
4. Cài Torch bản CPU-only cho VietOCR để tránh lỗi CUDA/NCCL trên Kaggle.
5. Cài:

```text
paddleocr
vietocr
scikit-learn
tqdm
pillow
opencv-python-headless
```

6. Re-pin Paddle GPU lần nữa phòng trường hợp pip resolver làm lệch version.

Lý do phải cẩn thận: Kaggle hay bị lỗi kiểu `libtorch_cuda.so: undefined symbol: ncclCommShrink` khi Torch CUDA và Paddle cùng tồn tại.

---

### Block 3–4 — Check version + GPU

Cell này import Paddle và PaddleOCR.

Có hàm quan trọng:

```python
safe_import_paddleocr(hide_torch=True)
```

Mục đích: khi import PaddleOCR, một số dependency có thể dò tìm Torch. Nếu Torch CUDA bị lỗi, import PaddleOCR cũng chết theo. Hàm này tạm thời “giấu” Torch khỏi `importlib.util.find_spec()` để PaddleOCR import được.

Sau đó notebook in ra:

- Python version
- Paddle version
- PaddleOCR version
- Paddle có thấy CUDA không
- GPU device hiện tại

---

### Block 5–6 — Load data và tìm ảnh test

Cell này tự dò dataset trong Kaggle hoặc local.

Các file quan trọng:

```text
test.csv
sample_submission.csv
test_images
train_labels.csv nếu có
```

Nó thử nhiều layout folder khác nhau, ví dụ:

```text
test_images/test_images/images
test_images/images
test_images
images
```

Nếu chưa extract ảnh, nó tìm zip như:

```text
test_images.zip
images.zip
```

rồi extract vào:

```text
/kaggle/working/dataset_images
```

Các biến output quan trọng:

```python
INPUT_DIR
WORK_DIR
IMAGES_DIR
TEST_CSV
SAMPLE_CSV
OUTPUT_CSV
CHECKPOINT_CSV
```

Trong đó:

```python
OUTPUT_CSV = /kaggle/working/submission.csv
CHECKPOINT_CSV = /kaggle/working/checkpoint_hybrid_paddle_vietocr_resnet.csv
```

---

### Block 7–8 — Product rules

Block này chứa nhiều `BRAND_RULES`.

Mỗi rule có dạng:

```python
(regex_pattern, canonical_product_name, product_line_keywords)
```

Ví dụ ý tưởng:

```text
OCR text có "ha long", "canfoco", "pate"
→ product_name = "Ha Long Canfoco Pate"
```

Rule được đặt rất cụ thể ở đầu để bắt các sản phẩm quan trọng trước, sau đó mới tới rule tổng quát.

---

### Block 9–10 — Train lightweight product predictor từ train labels

Cell này tạo product head từ `train_labels.csv`.

Nó có các hàm chính:

```python
normalize_match_text()
token_f1()
partial_char_ratio()
```

Mục đích:

- lowercase
- bỏ dấu tiếng Việt
- đổi `đ` thành `d`
- bỏ ký tự lạ
- so khớp fuzzy giữa OCR text và nhãn product

Nếu có `train_labels.csv`, nó fit một `product_predictor` dựa trên:

- phrase correction
- rule matching
- fuzzy matching
- TF-IDF/LogisticRegression được chuẩn bị nhưng bản final không cho fallback tự chọn nhãn bừa

Hàm chính ở cuối block:

```python
predict_product(ocr_text)
```

Ban đầu hàm này gọi `product_predictor.predict()` nếu predictor tồn tại.

---

### Block 11–12 — Final product policy: rule/fuzzy only, no LR fallback

Đây là block rất quan trọng.

Notebook ghi chú kết quả thực nghiệm:

```text
PaddleOCR GPU + LR fallback     : ~0.6717
PaddleOCR GPU + no-LR fallback  : ~0.6732
```

Vì vậy bản này tắt LogisticRegression fallback.

Biến mode:

```python
PRODUCT_MODE = "rule_fuzzy_no_lr_badlabel_guard"
```

Nó có `BAD_PRODUCT_PATTERNS`, ví dụ:

```text
coffee
highlands
music
breaking news
thuốc
acnes
...
```

Mục đích: nếu product dự đoán rơi vào những nhãn rõ ràng không thuộc domain Pate/Milk hoặc là noise TikTok, thì blank product lại.

Các route an toàn:

```python
SAFE_PRODUCT_ROUTES = {
    "rule_corrected",
    "rule_raw",
    "fuzzy_strong",
    "fuzzy_after_gate",
}
```

Ý nghĩa: chỉ giữ product nếu nó đến từ rule/fuzzy đủ chắc, không giữ nhãn do LR tự suy diễn yếu.

---

### Block 13–14 — PaddleOCR GPU engine

Đây là OCR chính.

Config:

```python
OCR_SCORE_THRESHOLD = 0.30
MAX_DIM = 1280
USE_SHARPEN = True
USE_CONTRAST = True
```

Ý nghĩa:

- `OCR_SCORE_THRESHOLD = 0.30`: nhận text có confidence từ 0.30 trở lên. Thấp hơn thì bắt được nhiều text hơn nhưng dễ nhiễu.
- `MAX_DIM = 1280`: resize cạnh dài nhất về tối đa 1280 để cân bằng tốc độ và độ chính xác.
- `USE_CONTRAST = True`: tăng contrast.
- `USE_SHARPEN = True`: sharpen ảnh.

Hàm chính:

```python
make_paddle_ocr()
```

Nó thử nhiều config PaddleOCR khác nhau để tương thích nhiều version.

Sau đó:

```python
paddle_reader = make_paddle_ocr(OCR_DEVICE)
```

Các hàm phụ:

```python
image_path_for_id(image_id)
load_image(image_id)
preprocess(img)
postprocess_ocr(text)
```

Luồng PaddleOCR cơ bản:

```text
image_id
  ↓
load image
  ↓
preprocess: resize + contrast + sharpen
  ↓
paddle_reader.predict()
  ↓
parse text + score
  ↓
postprocess text
```

---

### Block 15–16 — Hybrid OCR: Paddle base + VietOCR rescue

Đây là khác biệt lớn nhất của notebook này.

Config:

```python
USE_VIETOCR_RESCUE = True
VIETOCR_DEVICE = "cpu"
VIETOCR_MIN_QUALITY = 0.70
PADDLE_GOOD_MIN_AVG_SCORE = 0.38
PADDLE_MIN_CHARS_FOR_GOOD = 8
VIETOCR_MAX_CROPS = 20
VIETOCR_CROP_PAD_RATIO = 0.03
```

Ý nghĩa:

- VietOCR chạy CPU.
- Chỉ dùng VietOCR nếu PaddleOCR yếu.
- Text VietOCR phải có quality score >= 0.70 mới được thay PaddleOCR.
- Tối đa đọc 20 crop để tránh quá chậm.
- Crop text box được pad thêm 3%.

Hàm tạo VietOCR:

```python
make_vietocr_predictor()
```

Nó dùng config:

```python
vgg_transformer
```

và tắt beamsearch để nhanh hơn:

```python
cfg["predictor"]["beamsearch"] = False
```

#### Parser PaddleOCR

Hàm:

```python
parse_paddle_detail(output, score_threshold)
```

Nó cố đọc output từ nhiều format PaddleOCR khác nhau:

- PaddleOCR 3.x dict result
- legacy list output

Nó trả về:

```python
text, avg_score, lines, scores, polys
```

Trong đó `polys` là polygon text box dùng để crop cho VietOCR.

#### Đánh giá PaddleOCR có đủ tốt không

Hàm logic:

```python
is_good_paddle_text(paddle_text, paddle_avg)
```

Ý tưởng:

```text
Nếu avg_score đủ cao và text đủ dài → Paddle good
Nếu text có quality tốt/domain keyword → vẫn giữ Paddle
Ngược lại → cho VietOCR rescue
```

#### VietOCR rescue

Luồng:

```text
PaddleOCR output polys
  ↓
crop từng polygon text box
  ↓
VietOCR đọc từng crop
  ↓
ghép text
  ↓
quality gate
  ↓
nếu tốt hơn thì thay Paddle text
```

Các hàm:

```python
crop_poly_bbox(img, poly)
recognize_vietocr_on_polys(img, polys)
merge_paddle_vietocr(paddle_text, paddle_avg, viet_text)
```

#### Hàm cuối cùng override pipeline

Hàm quan trọng nhất:

```python
run_ocr(image_id)
```

Luồng của `run_ocr()`:

```text
1. load_image(image_id)
2. preprocess ảnh
3. chạy PaddleOCR
4. parse text, score, polygon
5. nếu Paddle tốt → giữ Paddle text
6. nếu Paddle yếu → crop polygon và cho VietOCR đọc lại
7. nếu VietOCR quality >= 0.70 → dùng VietOCR text
8. predict_product(final_text)
9. apply_resnet_gate(product_name, image_id, img)
10. return final_text, product_name
```

Lưu ý: VietOCR không tự tìm vùng chữ. Nó phụ thuộc vào polygon/crop từ PaddleOCR. Nếu PaddleOCR không detect được box nào, VietOCR thường cũng không có crop để cứu.

---

### Block 17–18 — Optional ResNet visual gate

Mặc định:

```python
USE_RESNET_GATE = False
```

Nó không tham gia final nếu bạn không bật.

Ý tưởng của nó:

```text
Image
  ↓
ResNet18
  ↓
domain prediction: none / pate / milk
  ↓
nếu product_name mâu thuẫn domain rất chắc → blank product
```

Config:

```python
RESNET_EPOCHS = 2
RESNET_BATCH_SIZE = 32
RESNET_GATE_CONF = 0.92
```

Classes:

```python
0 = none
1 = pate
2 = milk
```

Nó infer label train từ `product_name` bằng keyword:

```text
pate, ha long, canfoco, đồ hộp → pate
sữa, milk, nan, nestle, vinamilk → milk
```

Nếu bật, nó train ResNet18 bằng Paddle trên train images.

Khuyến nghị: **không bật cho final nếu chưa validate leaderboard**, vì gate sai có thể blank mất product đúng.

---

### Block 19–20 — Preview OCR

Config:

```python
PREVIEW_START = 0
PREVIEW_N = 20
```

Nó chạy thử `run_ocr()` trên một slice nhỏ rồi in:

```text
image_id
ocr_text
product_name
thời gian / ảnh
```

Dùng block này để kiểm tra pipeline có chạy thật không trước khi chạy full.

Muốn preview từ ảnh 700:

```python
PREVIEW_START = 700
PREVIEW_N = 20
```

---

### Block 21–22 — Full inference với checkpoint resume

Config:

```python
START_INDEX = 0
LIMIT = None
SAVE_EVERY = 25
FORCE_REOCR = False
```

Ý nghĩa:

- `START_INDEX = 0`: chạy từ đầu.
- `LIMIT = None`: chạy full test set.
- `SAVE_EVERY = 25`: cứ 25 ảnh lưu checkpoint một lần.
- `FORCE_REOCR = False`: nếu checkpoint đã có thì resume, không OCR lại.

Nếu muốn test nhanh 50 ảnh:

```python
START_INDEX = 0
LIMIT = 50
```

Nếu muốn chạy từ index 700:

```python
START_INDEX = 700
LIMIT = None
```

Nếu muốn bỏ checkpoint cũ và OCR lại từ đầu:

```python
FORCE_REOCR = True
```

Output tạm:

```text
/kaggle/working/checkpoint_hybrid_paddle_vietocr_resnet.csv
```

Checkpoint có cột:

```text
image_id, ocr_text, product_name
```

Sau khi chạy xong, notebook in:

```text
OCR fill
Product fill
OCR failures
```

---

### Block 23–24 — Validate và export `submission.csv`

Cell này build submission theo đúng thứ tự `sample_submission.csv`.

Nó fill ô trống bằng single space `" "` để hợp lệ với Kaggle.

Các check:

```text
AC-1 Row count match
AC-2 No duplicate IDs
AC-3 No null values
AC-4 Columns correct
AC-5 No newline/tab
AC-6 IDs match sample
```

Nếu pass, nó lưu:

```text
/kaggle/working/submission.csv
```

với encoding:

```text
utf-8
```

và quoting toàn bộ field bằng:

```python
csv.QUOTE_ALL
```

---

### Block 25–26 — Debug helpers

Cell này in các file trong `/kaggle/working` và các biến quan trọng:

```python
INPUT_DIR
IMAGES_DIR
TEST_CSV
SAMPLE_CSV
CHECKPOINT_CSV
OUTPUT_CSV
OCR_DEVICE
OCR_SCORE_THRESHOLD
MAX_DIM
product_predictor
```

Nó cũng display tail checkpoint và head submission nếu file tồn tại.

---

## 3. Cách chạy trên Kaggle

### Bước 1 — Tạo notebook Kaggle GPU

Trong Kaggle:

```text
Settings → Accelerator → GPU
Internet → On
```

Internet cần bật để pip install package.

---

### Bước 2 — Add data competition

Add dataset/competition data của URA vào notebook.

Notebook cần thấy ít nhất:

```text
test.csv
sample_submission.csv
test_images hoặc test_images.zip
```

Nếu có `train_labels.csv`, product predictor sẽ tốt hơn.

---

### Bước 3 — Chạy lần lượt từ trên xuống

Chạy theo thứ tự:

```text
Cell 1–2: install
Cell 3–4: version check
Cell 5–6: load data
Cell 7–12: product rules + product predictor
Cell 13–16: OCR engine + VietOCR rescue
Cell 17–18: ResNet gate, để mặc định OFF
Cell 19–20: preview
Cell 21–22: full inference
Cell 23–24: export submission
Cell 25–26: debug nếu cần
```

---

## 4. Cấu hình khuyến nghị cho final

Giữ như sau:

```python
USE_VIETOCR_RESCUE = True
VIETOCR_DEVICE = "cpu"
VIETOCR_MIN_QUALITY = 0.70

USE_RESNET_GATE = False

OCR_SCORE_THRESHOLD = 0.30
MAX_DIM = 1280
USE_SHARPEN = True
USE_CONTRAST = True

START_INDEX = 0
LIMIT = None
FORCE_REOCR = False
```

Không nên bật ResNet gate cho final nếu chưa có submit test riêng.

---

## 5. Cách chạy nhanh để test

### Test 20 ảnh preview

```python
PREVIEW_START = 0
PREVIEW_N = 20
```

Chạy cell preview.

### Test full inference 50 ảnh

Trong cell full inference:

```python
START_INDEX = 0
LIMIT = 50
```

Sau đó chạy full inference và export submission để check format.

---

## 6. Cách resume khi bị ngắt

Notebook tự resume từ:

```text
/kaggle/working/checkpoint_hybrid_paddle_vietocr_resnet.csv
```

Nếu bị ngắt, chỉ cần chạy lại các cell setup rồi chạy tiếp cell full inference.

Nó sẽ đọc checkpoint và bỏ qua ảnh đã xử lý:

```python
done_map = checkpoint
pending = target_ids not in done_map
```

---

## 7. Cách chạy lại từ đầu

Đổi:

```python
FORCE_REOCR = True
```

rồi chạy cell full inference.

Sau khi chắc chắn muốn resume bình thường lại, đổi về:

```python
FORCE_REOCR = False
```

---

## 8. Các lỗi thường gặp

### Lỗi PaddleOCR import do Torch/NCCL

Triệu chứng:

```text
libtorch_cuda.so: undefined symbol: ncclCommShrink
```

Notebook đã có `safe_import_paddleocr(hide_torch=True)` và cài Torch CPU-only để né lỗi này.

Nếu vẫn lỗi, restart runtime và chạy lại từ cell install.

---

### Paddle không thấy GPU

Kiểm tra cell version có:

```text
Paddle CUDA available: True
```

Nếu không, bật GPU trong Kaggle settings.

---

### VietOCR làm OCR nhiễu hơn

Tăng gate:

```python
VIETOCR_MIN_QUALITY = 0.80
```

hoặc tắt hẳn:

```python
USE_VIETOCR_RESCUE = False
```

---

### OCR quá chậm

Giảm:

```python
VIETOCR_MAX_CROPS = 10
```

hoặc tắt VietOCR rescue:

```python
USE_VIETOCR_RESCUE = False
```

---

### Product bị blank quá nhiều

Kiểm tra `BAD_PRODUCT_PATTERNS` và `SAFE_PRODUCT_ROUTES`.

Bản này cố tình conservative để tránh product hallucination. Nếu muốn product fill cao hơn, phải chấp nhận rủi ro sai nhãn.

---

## 9. Nên hiểu bản này như thế nào?

Bản này không phải train OCR mới hoàn toàn.

Nó là một pipeline hybrid:

```text
PaddleOCR: detector + recognizer chính
VietOCR: recognizer phụ, chỉ cứu khi Paddle yếu
Rule/Fuzzy: chuyển OCR text thành product_name
ResNet gate: optional visual sanity check
```

Điểm mạnh:

- Có checkpoint resume.
- Format submission chắc.
- Hạn chế LR hallucination.
- VietOCR được dùng conservative, giảm rủi ro phá text Paddle tốt.

Điểm yếu:

- VietOCR phụ thuộc box từ PaddleOCR. Nếu Paddle không detect box, VietOCR khó cứu.
- ResNet gate chưa bật mặc định vì có rủi ro chặn nhầm.
- Rule/fuzzy cần train labels và rule tốt; không phải model hiểu ngữ nghĩa thật.

---

## 10. File output cần nộp

File cuối cùng:

```text
/kaggle/working/submission.csv
```

Kaggle yêu cầu:

```text
2006 rows
header
columns: image_id, ocr_text, product_name
```

Notebook đã validate các điều kiện này trước khi lưu.

---

## 11. Nối VS Code `.ipynb` với Kaggle kernel bằng “VS Code Compatible URL”

Phần này đúng với trường hợp bạn đang mở file `.ipynb` trong VS Code và góc phải đang hiện kernel local, ví dụ:

```text
Python 3.13.13
```

Khi thấy như vậy, notebook đang chạy bằng Python local trên máy bạn. Muốn dùng GPU/môi trường Kaggle, bạn phải đổi kernel đó sang **Existing Jupyter Server** của Kaggle.

---

### 11.1. Hiểu đúng cơ chế

Luồng đúng là:

```text
VS Code local
  mở file .ipynb, chỉnh code, bấm Run cell
        ↓
VS Code Compatible URL
        ↓
Kaggle Jupyter Server
  chạy code thật trên GPU/môi trường Kaggle
        ↓
output trả về VS Code
```

Nghĩa là:

- File `.ipynb` có thể nằm trên máy bạn.
- Nhưng cell được execute trên Kaggle.
- Path trong code là path của Kaggle, ví dụ:

```text
/kaggle/input
/kaggle/working
```

chứ không phải path trên máy local.

---

### 11.2. Chuẩn bị trên VS Code

Cài extension:

```text
Python
Jupyter
```

Sau đó mở file:

```text
ura_hybrid_paddle_vietocr_resnet_gate.ipynb
```

trong VS Code.

Ở góc phải notebook, nếu thấy:

```text
Python 3.13.13
```

hoặc một kernel local khác, nghĩa là chưa nối Kaggle.

---

### 11.3. Chuẩn bị Kaggle Jupyter Server

Trên Kaggle:

1. Mở một Kaggle Notebook bất kỳ của competition hoặc tạo notebook mới.
2. Vào settings của Kaggle notebook.
3. Bật:

```text
Accelerator → GPU
Internet → On
```

4. Add data competition URA vào notebook:

```text
Add Data
```

Notebook Kaggle cần thấy dữ liệu trong:

```text
/kaggle/input
```

5. Trên menu Kaggle, chọn:

```text
Run → Kaggle Jupyter Server
```

6. Một panel bên phải sẽ hiện ra.
7. Bấm start server nếu Kaggle yêu cầu.
8. Copy dòng:

```text
VS Code Compatible URL
```

URL này thường đã chứa token xác thực. Không gửi URL này cho người khác.

---

### 11.4. Nối URL đó vào VS Code

Quay lại VS Code, trong file `.ipynb`:

1. Click vào kernel picker ở góc phải trên, chỗ đang hiện:

```text
Python 3.13.13
```

2. Chọn:

```text
Select Another Kernel
```

3. Chọn:

```text
Existing Jupyter Server
```

4. Chọn:

```text
Enter the URL of the running Jupyter Server
```

5. Paste nguyên dòng:

```text
VS Code Compatible URL
```

vừa copy từ Kaggle.

6. VS Code có thể hỏi đặt tên server. Đặt ví dụ:

```text
Kaggle URA GPU
```

7. Sau đó chọn kernel Python của Kaggle.

Nếu thành công, góc phải notebook sẽ không còn là Python local kiểu:

```text
Python 3.13.13
```

mà sẽ chuyển sang kernel từ Kaggle/Jupyter server.

---

### 11.5. Test xem cell có thật sự chạy trên Kaggle không

Chạy cell nhỏ này trong VS Code:

```python
import os, subprocess, sys

print("Python:", sys.version)
print("PWD:", os.getcwd())
print("Has /kaggle/input:", os.path.exists("/kaggle/input"))
print("Has /kaggle/working:", os.path.exists("/kaggle/working"))

subprocess.run("nvidia-smi", shell=True)
```

Kết quả đúng nên có:

```text
Has /kaggle/input: True
Has /kaggle/working: True
```

và `nvidia-smi` in ra GPU.

Nếu `/kaggle/input` là `False`, bạn vẫn đang chạy local hoặc Kaggle server chưa đúng notebook/session.

---

### 11.6. Chạy notebook OCR sau khi nối

Sau khi nối kernel thành công, chạy các cell theo thứ tự:

```text
1) Install PaddleOCR GPU + VietOCR
2) Version check + GPU check
3) Load data + auto-detect images
4) Product rules
5) Train product predictor
5.5) Final product head
6) PaddleOCR GPU engine
6.5) Hybrid OCR
6.6) Optional ResNet gate, để OFF
7) Preview OCR
8) Full OCR inference
9) Validate + export submission.csv
10) Debug helpers nếu cần
```

Vì code chạy trên Kaggle, output cuối vẫn nằm ở:

```text
/kaggle/working/submission.csv
```

---

### 11.7. Lưu ý cực quan trọng về file output

Khi bạn chạy từ VS Code nối Kaggle kernel:

```text
VS Code chỉ là nơi điều khiển
Kaggle là nơi chạy và lưu file
```

Nên file:

```text
submission.csv
checkpoint_hybrid_paddle_vietocr_resnet.csv
```

sẽ nằm trên Kaggle server:

```text
/kaggle/working/
```

không tự động xuất hiện trong folder local của VS Code.

Muốn lấy file về máy:

- dùng file browser/output panel của Kaggle để download, hoặc
- chạy lệnh trong notebook để copy/hiển thị link nếu môi trường hỗ trợ.

---

### 11.8. Nếu VS Code vẫn hiện Python local

Nếu góc phải vẫn là:

```text
Python 3.13.13
```

thì bạn chưa đổi kernel.

Làm lại:

```text
Click Python 3.13.13
→ Select Another Kernel
→ Existing Jupyter Server
→ Enter URL
→ paste VS Code Compatible URL
→ chọn kernel Kaggle
```

Nếu VS Code báo không connect được:

1. Kiểm tra Kaggle Jupyter Server còn chạy không.
2. Copy lại URL mới từ Kaggle.
3. Restart VS Code.
4. Đảm bảo extension Jupyter đã được cài.
5. Không paste thiếu token trong URL.

---

### 11.9. Nếu package install xong nhưng import lỗi

Vì notebook có cài lại Paddle/Torch/VietOCR, đôi khi cần restart kernel remote.

Làm trên Kaggle:

```text
Run → Restart Session
```

hoặc trong VS Code:

```text
Kernel → Restart Kernel
```

Sau đó chạy lại từ cell version check trở xuống.

Nếu reset session làm mất package, chạy lại cell install.

---

### 11.10. Checklist nối kernel thành công

Bạn nối đúng nếu:

```text
[ ] VS Code không còn chạy kernel local Python 3.13.13
[ ] /kaggle/input tồn tại
[ ] /kaggle/working tồn tại
[ ] nvidia-smi chạy được
[ ] Paddle CUDA available = True
[ ] OCR_DEVICE = gpu
```

Khi các dòng này ổn thì mới chạy full inference.

