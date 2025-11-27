# 🏦 Bank Statement Fraud Detection System

A comprehensive 6-layer fraud detection pipeline to identify tampering, forgery, and fraudulent behavior in bank statements using computer vision, OCR, and large language models.

## 🎯 Features

- **Multi-threaded Processing**: Concurrent analysis with thread-safe OCR engine pooling (4 workers)
- **6-Layer Detection Pipeline**: Comprehensive analysis from metadata to transaction-level fraud detection
- **Early Termination**: Smart filtering to skip expensive API calls on obviously fraudulent documents
- **Detailed Timing**: Layer-by-layer performance breakdown for optimization insights
- **Batch Processing**: Process multiple documents in parallel with consolidated reporting

## 🔬 Detection Layers

### 1️⃣ **Metadata Analysis**
- Extracts file creation/modification timestamps
- Detects high-risk editing software (Photoshop, iLovePDF, Smallpdf, Adobe Illustrator)
- Checks EXIF data for tampering indicators
- **Early termination** if professional editing software detected

### 2️⃣ **CLIP Verification** (OpenAI)
- Validates if uploaded file is actually a bank statement
- Identifies bank type and document format
- Detects document quality issues
- **Early termination** if not a bank statement

### 3️⃣ **ELA Detection** (Error Level Analysis)
- Identifies areas with different compression levels
- Detects potential copy-paste manipulations
- Skipped for PDF files

### 4️⃣ **Gemini Vision Analysis** (Google)
- AI-powered visual tampering detection
- Identifies inconsistencies in fonts, alignment, spacing
- Detects digital alterations and forgeries
- Confidence scoring with risk level assessment

### 5️⃣ **OCR Extraction** (PaddleOCR)
- Multi-threaded text extraction with independent engine pool
- Thread-safe round-robin engine allocation
- Extracts all text content for transaction analysis

### 6️⃣ **Gemini OCR Analysis** (Google)
- Transaction-level fraud pattern detection
- Identifies suspicious transactions and anomalies
- Detects unusual patterns (round numbers, duplicate amounts, etc.)
- Provides detailed fraud reasoning

## 📊 Risk Scoring

The system calculates a combined risk score (0.0 - 1.0) based on all layers:

- **ACCEPT** (`< 0.3`): Low risk, document appears genuine
- **MANUAL_REVIEW** (`0.3 - 0.6`): Medium risk, requires human verification
- **REJECT** (`> 0.6`): High risk, likely fraudulent

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+ required
pip install -r requirements.txt

# Install OpenAI CLIP (manual installation)
pip install git+https://github.com/openai/CLIP.git

# System dependencies (Windows)
# - Poppler: Download from https://github.com/oschwartz10612/poppler-windows/releases
```

### Environment Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Run Analysis

```bash
# Activate virtual environment
.\venv_native\Scripts\activate

# Run batch analysis on dataset folder
python src/app.py

# Or customize the folder path in src/app.py:
# folder_path = "./your_dataset_folder"
```

## 📁 Project Structure

```
FraudDetection/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── app.py                      # Main application (multi-threaded)
│   ├── app_backup.py              # Backup version with timing
│   │
│   ├── clip_module/               # CLIP-based verification
│   │   ├── __init__.py
│   │   └── clip_engine.py
│   │
│   ├── ela_detection/             # Error Level Analysis
│   │   ├── __init__.py
│   │   └── ela.py
│   │
│   ├── llm/                       # LLM-based analysis
│   │   ├── __init__.py
│   │   ├── gemini_vision.py      # Visual tampering detection
│   │   └── gemini_ocr.py         # Transaction fraud analysis
│   │
│   ├── ocr_test/                  # OCR processing
│   │   ├── __init__.py
│   │   ├── new_ocr.py            # Multi-threaded PaddleOCR
│   │   └── new_ocr_backup.py
│   │
│   └── metadata/                  # Metadata extraction
│       ├── __init__.py
│       └── metadata_analysis.py
│
├── dataset/                       # Input images/PDFs
├── ocr_results/                   # OCR output JSON files
├── results_record_flash.json      # Batch analysis results
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

### Threading Configuration

```python
# src/app.py
max_workers = 4  # Number of parallel threads

# src/ocr_test/new_ocr.py
_ocr_parallel_count = 4  # OCR engine pool size
```

### Detection Thresholds

```python
# src/app.py - FraudDetectionSystem.__init__()
ela_threshold = 30.0      # ELA sensitivity
use_gemini = True         # Enable Gemini Vision
use_ocr = True            # Enable OCR + Gemini OCR
```

### Early Termination

High-risk editing software (automatic rejection):
- Adobe Photoshop
- iLovePDF
- Smallpdf
- Adobe Illustrator

## 📈 Performance

**Typical Processing Time** (4 workers):
- OCR Extraction: ~22% of total time
- Gemini Vision: ~43% of total time
- Gemini OCR: ~32% of total time

**Throughput**: ~4x faster than sequential processing

## 🔧 Troubleshooting

### Common Issues

**"Tensor holds no memory" error**:
- Fixed by using independent OCR engine pool (one per thread)

**CLIP import conflict**:
- Renamed `src/clip` to `src/clip_module` to avoid conflict with OpenAI CLIP library

**Gemini API rate limits**:
- Free tier: 15 RPM
- Adjust `sleep_time` in `_analyze_single_safe_capture()` if hitting limits

## 📝 Output

Results are saved to `results_record_flash.json` with:
- Timestamp
- Per-file recommendations (ACCEPT/MANUAL_REVIEW/REJECT)
- Execution time breakdown
- Batch statistics

## 🛡️ Security Features

- ✅ Early termination prevents wasted API calls on fraudulent documents
- ✅ Metadata-based software detection catches professional forgeries
- ✅ Multi-layer verification reduces false negatives
- ✅ Thread-safe processing ensures data integrity

## 📄 License

MIT License

## 👤 Author

JF_EVVO

## 🙏 Acknowledgments

- OpenAI CLIP for document verification
- Google Gemini for AI-powered analysis
- PaddleOCR for text extraction
- PyMuPDF for PDF processing