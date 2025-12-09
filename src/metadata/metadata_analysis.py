"""
Test: Metadata extraction for images and PDFs
"""
import os, json, fitz
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime

def get_file_info(path: str) -> dict:
    """Extract file system metadata"""
    try:
        stat = os.stat(path)
        created = datetime.fromtimestamp(stat.st_ctime)
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        return {
            'filename': Path(path).name,
            'created': created.strftime('%Y-%m-%d %H:%M:%S'),
            'modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
            'time_diff_seconds': (modified - created).total_seconds()
        }
    except Exception as e:
        return {'error': str(e)}

def extract_pdf_metadata(pdf_path: str) -> dict:
    """Extract PDF metadata"""
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        result = {
            'file_info': get_file_info(pdf_path),
            'pdf_metadata': {
                'author': meta.get('author', ''),
                'creator': meta.get('creator', ''),
                'producer': meta.get('producer', ''),
                'creationDate': meta.get('creationDate', ''),
                'modDate': meta.get('modDate', '')
            },
            'file_type': 'PDF'
        }
        doc.close()
        return result
    except Exception as e:
        return {'file_info': get_file_info(pdf_path), 'pdf_metadata': {'error': str(e)}, 'file_type': 'PDF'}

def extract_metadata(path: str) -> dict:
    """Extract metadata from image or PDF"""
    if path.lower().endswith('.pdf'):
        return extract_pdf_metadata(path)
    
    metadata = {'file_info': get_file_info(path), 'exif_data': {}, 'file_type': 'IMAGE'}
    
    try:
        img = Image.open(path)
        
        exif = img._getexif()
        if exif:
            for tag_id, val in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Only keep Software field for editing software detection
                if tag_name == 'Software':
                    metadata['exif_data'][tag_name] = val.decode('utf-8', errors='ignore') if isinstance(val, bytes) else val
        
        if not metadata['exif_data']:
            metadata['exif_data']['note'] = 'No Software info found'
    except Exception as e:
        metadata['exif_data']['error'] = str(e)
    
    return metadata

def check_tampering_indicators(metadata: dict) -> list:
    """Check for tampering indicators - returns list of flags"""
    flags = []
    file_type = metadata.get('file_type', 'IMAGE')
    
    if file_type == 'PDF':
        pdf = metadata.get('pdf_metadata', {})
        edit_sw = ['Photoshop', 'GIMP', 'Acrobat', 'PDFtk', 'iLovePDF', 'PDFium', 'Sejda', 'Smallpdf', 'Adobe Illustrator']
        creator, producer = pdf.get('creator', ''), pdf.get('producer', '')
        
        for sw in edit_sw:
            if sw.lower() in creator.lower() or sw.lower() in producer.lower():
                # Mark PDFium separately for lower risk score
                if sw.lower() == 'pdfium':
                    flags.append(f'pdf_editing_software_low_risk: {sw}')
                else:
                    flags.append(f'pdf_editing_software: {sw}')
        
        if pdf.get('creationDate') and pdf.get('modDate') and pdf['creationDate'] != pdf['modDate']:
            flags.append('pdf_modified')
        if not creator and not producer:
            flags.append('pdf_missing_metadata')
    else:
        exif = metadata.get('exif_data', {})
        edit_sw = ['Photoshop', 'GIMP', 'Affinity', 'Paint.NET', 'Pixlr']
        software = exif.get('Software', '')
        
        for sw in edit_sw:
            if sw.lower() in str(software).lower():
                flags.append(f'image_editing_software: {sw}')
        
        fi = metadata.get('file_info', {})
        if 'time_diff_seconds' in fi and fi['time_diff_seconds'] > 60:
            flags.append(f'file_modified ({fi["time_diff_seconds"]:.0f}s after creation)')
    
    return flags

def main():
    """Test metadata extraction"""
    folder = Path("./dataset")
    paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.pdf']:
        paths.extend(folder.glob(ext))
        paths.extend(folder.glob(ext.upper()))
    
    paths = sorted(set(str(p) for p in paths))
    
    if not paths:
        print(f"❌ No files found in {folder}")
        return
    
    print(f"Found {len(paths)} files to analyze\n")
    
    for path in paths:
        if not os.path.exists(path):
            print(f"\n❌Not found: {path}")
            continue
        
        print(f"\n{'='*80}\nANALYZING: {Path(path).name}\n{'='*80}")
        
        meta = extract_metadata(path)
        print(f"\nFile Info:")
        for k, v in meta['file_info'].items():
            print(f"  {k}: {v}")
        
        if meta.get('file_type') == 'PDF':
            print(f"\nPDF Metadata:")
            for k, v in meta.get('pdf_metadata', {}).items():
                print(f"  {k}: {v}")
        else:
            # Only show Software field if exists
            software = meta['exif_data'].get('Software', meta['exif_data'].get('note', 'N/A'))
            print(f"\nSoftware: {software}")
        
        flags = check_tampering_indicators(meta)
        print(f"\nTampering: {len(flags) > 0}")
        if flags:
            for flag in flags:
                print(f"  ⚠️  {flag}")
        else:
            print(f"  ✅ No indicators")
        
        output_dir = Path("./extracted_data")
        output = output_dir / f"metadata_{Path(path).stem}.json"
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved: {output}")

if __name__ == "__main__":
    main()
