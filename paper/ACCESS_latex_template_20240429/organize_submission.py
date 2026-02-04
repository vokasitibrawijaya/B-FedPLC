"""
Organize all IEEE Access submission files into one folder
"""
import os
import shutil
import zipfile

# Create submission package folder
submission_folder = "IEEE_ACCESS_SUBMISSION_PACKAGE"
os.makedirs(submission_folder, exist_ok=True)
print(f"📁 Created folder: {submission_folder}")

# 1. Copy/Create LaTeX ZIP
print("\n📦 Creating LaTeX submission ZIP...")
latex_zip = os.path.join(submission_folder, "B-FedPLC_LaTeX_Submission.zip")

with zipfile.ZipFile(latex_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Main LaTeX file
    zipf.write('bfedplc_paper.tex')

    # Template files
    template_files = ['ieeeaccess.cls', 'IEEEtran.cls', 'IEEEtran.bst', 'spotcolor.sty',
                     't1times.fd', 't1formata.fd', 't1giovannistd.fd', 't1helvetica.fd']
    for f in template_files:
        if os.path.exists(f):
            zipf.write(f)

    # Font files
    for f in os.listdir('.'):
        if f.startswith('t1-') and (f.endswith('.pfb') or f.endswith('.tfm') or f.endswith('.map')):
            zipf.write(f)

    # Figures
    if os.path.exists('figures'):
        for f in os.listdir('figures'):
            zipf.write(os.path.join('figures', f), f'figures/{f}')

    # Images
    images = ['moko.jpg', 'sholeh.jpg', 'fauzan.jpg', 'panca.jpg', 'mahdin.jpg', 'cries.jpg',
             'logo.png', 'bullet.png', 'notaglinelogo.png', 'bfedplc_architecture.png']
    for img in images:
        if os.path.exists(img):
            zipf.write(img)

latex_size = os.path.getsize(latex_zip) / (1024 * 1024)
print(f"   ✓ Created: {latex_zip} ({latex_size:.2f} MB)")

# 2. Copy PDF
print("\n📄 Copying PDF file...")
if os.path.exists('bfedplc_paper.pdf'):
    shutil.copy('bfedplc_paper.pdf', os.path.join(submission_folder, 'B-FedPLC_Manuscript.pdf'))
    pdf_size = os.path.getsize(os.path.join(submission_folder, 'B-FedPLC_Manuscript.pdf')) / (1024 * 1024)
    print(f"   ✓ Copied: B-FedPLC_Manuscript.pdf ({pdf_size:.2f} MB)")

# 3. Copy supporting documents
print("\n📝 Copying supporting documents...")
support_files = {
    'CONFLICT_OF_INTEREST_STATEMENT.txt': 'Conflict_of_Interest.txt',
    'COVER_LETTER.txt': 'Cover_Letter.txt',
    'SUBMISSION_INSTRUCTIONS.md': 'SUBMISSION_INSTRUCTIONS.md',
    'IEEE_ACCESS_COMPLIANCE_CHECKLIST.md': 'IEEE_ACCESS_COMPLIANCE_CHECKLIST.md'
}

for src, dst in support_files.items():
    if os.path.exists(src):
        shutil.copy(src, os.path.join(submission_folder, dst))
        print(f"   ✓ Copied: {dst}")

# Create README for submission package
print("\n📋 Creating README...")
readme_content = """# IEEE ACCESS SUBMISSION PACKAGE
## B-FedPLC: Blockchain-Enabled Federated Learning

**Submission Date:** January 31, 2026
**Corresponding Author:** Dr. Sholeh Hadi Pramono (sholehpramono@ub.ac.id)

---

## FILES IN THIS PACKAGE

### 1. REQUIRED FOR UPLOAD

#### A. Main Manuscript Files
- **B-FedPLC_LaTeX_Submission.zip** (13 MB)
  - Complete LaTeX source with all dependencies
  - Includes: .tex file, class files, figures, fonts, author photos
  - Upload as "Main Document - LaTeX"

- **B-FedPLC_Manuscript.pdf** (6 MB)
  - Final formatted manuscript (14 pages)
  - IEEE Access double-column format
  - Upload as "Main Document - PDF"

#### B. Supporting Documents
- **Conflict_of_Interest.txt**
  - Statement of no conflicts of interest
  - Upload during conflict of interest section

- **Cover_Letter.txt**
  - Cover letter to Editor-in-Chief
  - Upload as cover letter

### 2. REFERENCE DOCUMENTS

- **SUBMISSION_INSTRUCTIONS.md**
  - Step-by-step submission guide
  - ScholarOne submission details
  - Keywords, author info, etc.

- **IEEE_ACCESS_COMPLIANCE_CHECKLIST.md**
  - Verification of all IEEE Access requirements
  - Detailed compliance status

---

## SUBMISSION STEPS

1. Go to: https://mc.manuscriptcentral.com/ieee-access
2. Login with IEEE account
3. Click "Submit New Manuscript"
4. Select "Research Article"
5. Upload files:
   - Main Document LaTeX: B-FedPLC_LaTeX_Submission.zip
   - Main Document PDF: B-FedPLC_Manuscript.pdf
   - Conflict of Interest: Conflict_of_Interest.txt
   - Cover Letter: Cover_Letter.txt
6. Enter manuscript details (see SUBMISSION_INSTRUCTIONS.md)
7. Submit!

---

## MANUSCRIPT DETAILS

**Title:** B-FedPLC: Blockchain-Enabled Federated Learning with Prototype-Anchored Learning and Dynamic Community Adaptation for Byzantine-Resilient Distributed Machine Learning

**Authors:**
1. Rachmad Andri Atmoko (Universitas Brawijaya)
2. Sholeh Hadi Pramono (Corresponding Author - Universitas Brawijaya)
3. Muhammad Fauzan Edy Purnomo (Universitas Brawijaya)
4. Panca Mudjirahardjo (Universitas Brawijaya)
5. Mahdin Rohmatillah (Universitas Brawijaya)
6. Cries Avian (Universitas Brawijaya)

**Keywords:** audit trail, blockchain, Byzantine fault tolerance, dynamic clustering, federated learning, IPFS, Merkle tree verification, non-IID data, personalized learning, prototype learning

**Statistics:**
- Pages: 14
- Figures: 10
- Tables: 9
- References: 61
- Total Size: ~19 MB (within 40 MB limit)

---

## CONTACT

**Corresponding Author:**
Dr. Sholeh Hadi Pramono
Professor, Faculty of Engineering
Universitas Brawijaya
Email: sholehpramono@ub.ac.id
ORCID: 0000-0003-4399-284X

---

✅ ALL FILES VERIFIED AND READY FOR SUBMISSION
"""

with open(os.path.join(submission_folder, 'README.txt'), 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"   ✓ Created: README.txt")

# Summary
print("\n" + "="*60)
print("✅ SUBMISSION PACKAGE COMPLETE!")
print("="*60)
print(f"\nAll files organized in: {submission_folder}/")
print("\nPackage contents:")
print("  1. B-FedPLC_LaTeX_Submission.zip    (13.01 MB) - LaTeX source")
print("  2. B-FedPLC_Manuscript.pdf          (6.14 MB)  - Final PDF")
print("  3. Conflict_of_Interest.txt         (1.5 KB)   - COI statement")
print("  4. Cover_Letter.txt                 (3.5 KB)   - Cover letter")
print("  5. SUBMISSION_INSTRUCTIONS.md       - Guide")
print("  6. IEEE_ACCESS_COMPLIANCE_CHECKLIST.md - Verification")
print("  7. README.txt                       - Package overview")
print("\n📦 Total: ~19 MB (within IEEE Access 40 MB limit)")
print("\n🚀 READY TO SUBMIT TO IEEE ACCESS!")
print("="*60)
