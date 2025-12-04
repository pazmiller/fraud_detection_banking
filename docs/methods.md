# Bank Statement Fraud Detection - Technical Methods Documentation

## Method 1: Metadata Analysis

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

## Method 2: CLIP Analysis

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
