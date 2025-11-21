import os
import base64
import fitz  
from dotenv import load_dotenv
from groq import Groq


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

pdf_file_path = r"./dataset/ocbc_bank_statement_ps.jpg"  

def get_all_pages_as_images(pdf_path):
    image_list = []
    if pdf_path.endswith('.pdf'):
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            print(f" It has {total_pages} 页")

            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                pix = page.get_pixmap(matrix=fitz.Matrix()) 
                image_bytes = pix.tobytes("png")
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
                
                image_list.append({
                    "page_num": page_num + 1,
                    "base64": base64_str
                })
                
            doc.close()
            return image_list
        except Exception as e:
            print(f"PDF conversion error: {e}")
            return []
    else: # If its not PDF then dont convert l
        try:
            with open(pdf_path, 'rb') as f:
                image_bytes = f.read()
                base64_str = base64.b64encode(image_bytes).decode('utf-8')
                
                image_list.append({
                    "page_num": 1,
                    "base64": base64_str
                })
            return image_list
        except Exception as e:
            print(f"Image reading error: {e}")
            return []

pages_data = get_all_pages_as_images(pdf_file_path)

if pages_data:
    print(f"发 {len(pages_data)} pages of data for analysis")
    content_payload = [
        {
            "type": "text",
            "text": (
               """You are a highly sensitive Compliance Officer responsible for validating bank statements. Your goal is to identify DEFINITIVE evidence of document tampering while ignoring artifacts caused by scanning, printing, or standard PDF generation.

                    Real documents often have minor rendering imperfections. Only flag an issue if it is blatant and unexplainable.

                    First, determine the document type:
                    1. Digital Native: Created directly by software.
                    2. Scanned/Photo: A picture of a physical paper (noise, rotation, blur allowed).
                    ---
                    #ANALYSIS CRITERIA
                    1. Visual Consistency
                    -If the background "noise" or texture suddenly disappears behind specific numbers (indicating a digital patch).
                    -Look at the empty space SURROUNDING the transaction numbers. Does the texture/noise pattern suddenly
                    become "flat" or "white" behind a specific number, while the rest of the page has paper grain or digital noise?
                    - Out of place smudges, ink, irregular colour
                    2. Alignment & Layout
                    -If a specific number clearly floats outside its column grid while neighbors are aligned.
                    3. Artifacts & Compression**
                    - "Ghosting" or "halos" that appear ONLY around the transaction amount but nowhere else.
                    #EXCLUSION LIST
                    -Global blurriness.
                    -Math.
                    -Standard variation in font weight (Bold/Normal) used for emphasis.
                    -Imperfect letter spacing in PDF generation.
                    -Column text that is left-aligned vs. right-aligned numbers (standard accounting format).
                    -Different fonts used for Headers vs. Body vs. Footers.

                    ### RESPONSE FORMAT

                    Provide your analysis in this exact format:

                    TAMPERING_DETECTED: [YES/NO]
                    CONFIDENCE: [0-100]%
                    RISK_LEVEL: [LOW/MEDIUM/HIGH]
                    Type: [Digital Native / Scanned]

                    FINDINGS:
                    - [If none, state "No meaningful anomalies detected."]

                    RECOMMENDATION: [ACCEPT/MANUAL_REVIEW/REJECT]
"""
            )
        }
    ]

    for page in pages_data:
        content_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page['base64']}",
            },
        })

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content_payload,
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=2048, 
        )

        result = chat_completion.choices[0].message.content
        
        print(result)
        print("="*80)


    except Exception as e:
        print(f"API request fail: {e}")

else:
    print("CConversion fail")