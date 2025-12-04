# Bank Statement Fraud Detection - Technical Methods Documentation

## Method/Layer 1: Metadata Analysis

**Type:** Rule-based file metadata analysis  
**Purpose:** Detect editing software and file modification patterns

**Detection Criteria:**
- PDF editing software: `Photoshop`, `GIMP`, `Acrobat`, `PDFtk`, `iLovePDF`, `Sejda`, `Smallpdf`, `Adobe Illustrator`
- Image editing software: `Photoshop`, `GIMP`, `Affinity`, `Paint.NET`, `Pixlr`
- File modification time > 60 seconds after creation

**Early Termination:**
- If high-risk software detected (`Photoshop`, `iLovePDF`, `Smallpdf`, `Adobe Illustrator`), immediately REJECT without further analysis

**Risk Contribution:**
- Time difference > 100s: `+0.2` to the risk score

---

## Method/Layer 2: CLIP Analysis

**Type:** Neural network-based image-text analysis model  
**Model:** OpenAI CLIP `ViT-B/16` 

**Text Prompts:**
```python
# Document Type
["a bank statement", "receipt", "fake document", "handwritten"]
```

**Decision Rule:**
- If document is NOT classified as "bank statement" → Early termination, REJECT

**Key Parameters:**
- `confidence_threshold`: 0.3
- 'Device: CUDA' if available, else CPU (The model is tested on CPU for compatibility as non-Nvidia GPUs are common in deployment environments)

---

## Method/Layer 3: ELA (Error Level Analysis)

**Type:** Image forensics algorithm  
**Purpose:** Detect JPEG re-compression artifacts indicating manipulation

**Key Parameters:**
- `threshold`: 30.0 (default)
- `quality`: 90 (JPEG re-compression)

**Risk Contribution:**
- If suspicious: `+0.15`

**Notes:**
- Skipped for PDF and PNG files (not applicable)

---

## Method/Layer 4: LLM Vision Analysis

**Type:** Multimodal LLM visual analysis  
**Model Options:**
- `gemini-2.5-pro` (Google)
- `google/gemma-3-27b-it:free` (via OpenRouter)

**Prompt Focus:**
1. Visual Consistency - Background noise/texture anomalies
2. Alignment & Layout - Text positioning irregularities
3. Artifacts & Compression - Ghosting/halos around specific areas

**Output Format:**
```
TAMPERING_DETECTED: [YES/NO]
CONFIDENCE: [0-100]% (How sure the model is about the tampering, both YES and NO)
RISK_LEVEL: [LOW/MEDIUM/HIGH] (based on risk score)
FINDINGS: [list]
RECOMMENDATION: [ACCEPT/MANUAL_REVIEW/REJECT] (Both MANUAL_REVIEW and REJECT require human intervention at this stage)
```

**Key Parameters:**
- `temperature`: 0.1
- `max_completion_tokens`: 1024

**Risk Contribution:**
- If tampering detected: `+ confidence × 0.3`
- If recommendation = REJECT: set minimum risk to 0.6

---

## Method/Layer 5: OCR Extraction 
**Type:** OCR Text extraction (Optical Character Recognition)  
**Model:** PaddleOCR v2.7.x (< 3.0) (As currently latest 3.0 version has stability issues)

**Key Parameters:**
```python
use_angle_cls=True
lang='en' # ch for Bilingual Chinese+English, which might increase accuracy on certain cases but its more resource heavy and might result in tensor memory leakage
det_db_box_thresh=0.3
rec_batch_num=6
use_space_char=True
use_gpu=False
det_limit_side_len=960
```

**Output:**
- JSON file with `text_lines` and `full_text`, for LLM OCR analyser to analyse in the next layer

---

## Method 6/Layer 5: LLM OCR Analysis

**Type:** LLM-based transaction fraud detection, text analysis  
**Model Options:**
- `gemini-2.5-pro` (Google)
- `google/gemma-3-27b-it:free` (via OpenRouter)

**Analysis Focus:**
1. **Transaction Pattern Anomalies**

2. **Numerical (Balance Issues)**

3. **Formatting Issues (Inconsistence and Alterations)**


**Output Format:**
```
FRAUD_DETECTED: [YES/NO]
RISK_LEVEL: [LOW/MEDIUM/HIGH/CRITICAL]
SUSPICIOUS_TRANSACTIONS: [list]
PATTERNS_DETECTED: [list]
RECOMMENDATION: [ACCEPT/MANUAL_REVIEW/REJECT]
```

**Key Parameters:**
- `temperature`: 0.1
- `max_completion_tokens (LLM OCR)`: 2048
- `max_completion_tokens (LLM Vision)`: 1024

**Risk Contribution:**
- If fraud detected: `+ confidence × 0.4`
- If recommendation = REJECT: set minimum risk to 0.7

---

## Overall Risk Scoring System

**Final Risk Calculation:**
```python
risk = clip_base_risk

# Add contributions from each layer
if ela_suspicious: risk += 0.15
if clip_tampering > 0.3: risk = max(risk, 0.3)
if metadata_time_diff > 100: risk += 0.2
if metadata_editing_software: risk += 0.3
if vision_tampering: risk += confidence × 0.3
if vision_reject: risk = max(risk, 0.6)
if ocr_fraud: risk += confidence × 0.4
if ocr_reject: risk = max(risk, 0.7)

# Cap at 1.0
risk = min(risk, 1.0)
```

**Decision Thresholds:**
| Risk Score | Risk Level | Recommendation |
|------------|------------|----------------|
| 0.0 - 0.3  | LOW        | ACCEPT         |
| 0.3 - 0.6  | MEDIUM     | MANUAL_REVIEW  |
| 0.6 - 1.0  | HIGH       | REJECT         |
| Special    | CRITICAL   | REJECT (Early Termination) |

---

## Early Termination Conditions

1. **Metadata Stage:** High-risk editing software detected
   - Software: `Photoshop`, `iLovePDF`, `Smallpdf`, `Adobe Illustrator`
   - Result: CRITICAL, REJECT

2. **CLIP Stage:** Document not classified as bank statement
   - Result: CRITICAL, REJECT

---
---

## Performance Notes
- **Parallel Processing:** 4 concurrent threads (ThreadPoolExecutor)
- **Rate Limiting:** Random jitter (0.5-1.5s) + 1s delay after Vision API
- **Typical Timing Breakdown:**
  - OCR Extraction: ~22%
  - LLM Vision: ~43%
  - LLM OCR: ~32%

---

## Version History
1.0.1