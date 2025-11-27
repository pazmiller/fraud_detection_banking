# Bank Statement Fraud Detection System

A comprehensive 6-layer fraud detection pipeline to identify tampering, forgery, and fraudulent behavior in bank statements using computer vision, OCR, and large language models.

Layer structure:
1: Metadata Analysis
2: CLIP (Powered by OpenAI)
3: ELA
4: LLM Tampering Vision Analysis (Powered by Gemini)
5: OCR Extraction + LLM OCR Analysis (Powered by Gemini)

# Features
- **Multi-threaded Processing**: Concurrent with invididual multi-threading OCR processes (4 threads by default)
- **5-Layer Comprehensive Fraud Detection**: Comprehensive analysis from metadata to transaction-level fraud detection. It is able to achieve more than around 90% accuracy on Gemini-2.5-Pro and around 85% accuracy on Gemini-2.5-Flash
- **Early Termination**: Smart filtering to save time and to skip expensive API calls on obviously fraudulent documents, at Metadata, and CLIP stages.
- **Batch Processing**: Process multiple documents in paralle

## Detection Layers

# 1️⃣ **Metadata Analysis**
- Extracts file creation/modification timestamps as 
- Detects high-risk editing software (Photoshop, iLovePDF, Smallpdf,etc.) from EXIF data
- **Early termination** if professional editing software detected

# 2️⃣ **CLIP Verification** (OpenAI)
- Validates if uploaded file is actually a bank statement, as wrong documents could be uploaded by mistakes
- **Early termination** if not a bank statement

# 3️⃣ **ELA Detection** (Error Level Analysis)
- Identifies areas with different compression levels to detect potential tampering
- Skipped for PDF files due to ELA;s fundamental limitations on PDF files

# 4️⃣ **Gemini Vision Analysis** (Google Gemini)
- LLM-powered visual tampering detection in case previous three could not detect tampering
- Identifies inconsistencies in fonts, alignment, spacing
- Detects digital alterations and forgeries
- Confidence scoring with risk level assessment

# 5️⃣ **OCR Extraction** (PaddleOCR)
- Multi-threaded text extraction with independent engine pool (to avoid tensor memory overflow)
- Extracts all text content for next stage's LLM transaction analysis

# **Gemini OCR Analysis** (Google Gemini)
- Transaction-level fraud pattern detection
- Identifies suspicious transactions and anomalies
- Detects unusual patterns (round numbers, duplicate amounts, etc.)
- Provides detailed fraud reasoning

## Risk Scoring System

The system calculates a combined risk score (0.0 - 1.0) based on all layers:


# Quick Start Prerequisites

```bash
# Python 3.9+ required
pip install -r requirements.txt
# Install OpenAI CLIP (manual installation)
pip install git+https://github.com/openai/CLIP.git

### Environment Setup

Create a `.env` file in the project root:
GEMINI_API_KEY=your_gemini_api_key_here


```
### Run Analysis
# python src/app.py to Run the pipeline

# Run batch analysis on dataset folder
default ./dataset folder is used
Or customize the folder path in src/app.py:
folder_path = "./your_dataset_folder"



## ⚙️ Configuration
```python
# src/app.py
max_workers = 4  # Number of parallel threads

# src/ocr_test/new_ocr.py
_ocr_parallel_count = 4  # OCR engine pool size
```


## Troubleshooting for Common Issues

**"Tensor holds no memory" error**:
- Fixed by using independent OCR engine pool (one per thread)

**CLIP import conflict**:
- Renamed `src/clip` to `src/clip_module` to avoid conflict with OpenAI CLIP library

**Gemini API rate limits**:
- Recommended to use Paid Tier APIs, as Free tier's limit of: 15 RPM could result in rejected API requests

## Output

Results are saved to `results_record_flash.json` with:
- Timestamp
- Per-file recommendations (ACCEPT/MANUAL_REVIEW/REJECT)
- Execution time breakdown
- Batch statistics

## Documen recjection criteria (Fraud Verdict)
- Final risk score > 0.6
- Not a bank statement (CLIP)
- Edited with suspicious software (Metadata)
- ELA > 30
- time_different of file creation > 100 seconds