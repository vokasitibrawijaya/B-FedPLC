import os
import shutil
import zipfile

print("="*70)
print("ORGANIZING IEEE ACCESS SUBMISSION PACKAGE")
print("="*70)

# Define paths
source_dir = r"C:\Users\ADMIN\Documents\project\disertasis3\B-FedPLC-Blockchain Enable FL Dynamic Cluster\paper\ACCESS_latex_template_20240429"
dest_dir = r"C:\Users\ADMIN\Documents\project\disertasis3\B-FedPLC-Blockchain Enable FL Dynamic Cluster\paper\IEEE_ACCESS_SUBMISSION"

# Create destination folder
os.makedirs(dest_dir, exist_ok=True)
print(f"\n✓ Created folder: {dest_dir}")

# 1. Create LaTeX ZIP
print("\n[1/5] Creating LaTeX submission ZIP...")
latex_zip_path = os.path.join(dest_dir, "B-FedPLC_LaTeX_Submission.zip")

os.chdir(source_dir)

with zipfile.ZipFile(latex_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Main files
    files_to_zip = ['bfedplc_paper.tex', 'ieeeaccess.cls', 'IEEEtran.cls',
                    'IEEEtran.bst', 'spotcolor.sty']
    for f in files_to_zip:
        if os.path.exists(f):
            zipf.write(f)
            print(f"  + {f}")

    # Figures
    if os.path.exists('figures'):
        for f in os.listdir('figures'):
            zipf.write(os.path.join('figures', f), f'figures/{f}')
        print(f"  + figures/ ({len(os.listdir('figures'))} files)")

    # Author photos
    photos = ['moko.jpg', 'sholeh.jpg', 'fauzan.jpg', 'panca.jpg', 'mahdin.jpg', 'cries.jpg']
    for photo in photos:
        if os.path.exists(photo):
            zipf.write(photo)
    print(f"  + author photos (6 files)")

    # Logo images
    logos = ['logo.png', 'bullet.png', 'notaglinelogo.png', 'bfedplc_architecture.png']
    for logo in logos:
        if os.path.exists(logo):
            zipf.write(logo)
    print(f"  + logos and architecture diagram")

zip_size = os.path.getsize(latex_zip_path) / (1024 * 1024)
print(f"✓ Created: B-FedPLC_LaTeX_Submission.zip ({zip_size:.2f} MB)")

# 2. Copy PDF
print("\n[2/5] Copying PDF manuscript...")
pdf_src = os.path.join(source_dir, 'bfedplc_paper.pdf')
pdf_dest = os.path.join(dest_dir, 'B-FedPLC_Manuscript.pdf')
shutil.copy2(pdf_src, pdf_dest)
pdf_size = os.path.getsize(pdf_dest) / (1024 * 1024)
print(f"✓ Copied: B-FedPLC_Manuscript.pdf ({pdf_size:.2f} MB)")

# 3. Copy supporting documents
print("\n[3/5] Copying supporting documents...")
docs = {
    'CONFLICT_OF_INTEREST_STATEMENT.txt': 'Conflict_of_Interest.txt',
    'COVER_LETTER.txt': 'Cover_Letter.txt',
    'SUBMISSION_INSTRUCTIONS.md': 'SUBMISSION_INSTRUCTIONS.md',
    'IEEE_ACCESS_COMPLIANCE_CHECKLIST.md': 'IEEE_ACCESS_COMPLIANCE_CHECKLIST.md'
}

for src_name, dest_name in docs.items():
    src_path = os.path.join(source_dir, src_name)
    dest_path = os.path.join(dest_dir, dest_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        print(f"✓ Copied: {dest_name}")

# 4. Create README
print("\n[4/5] Creating README...")
readme_content = """# IEEE ACCESS SUBMISSION PACKAGE
## B-FedPLC: Blockchain-Enabled Federated Learning

Submission Date: January 31, 2026
Corresponding Author: Dr. Sholeh Hadi Pramono (sholehpramono@ub.ac.id)

=================================================================
FILES FOR UPLOAD TO IEEE SCHOLARONE
=================================================================

1. B-FedPLC_LaTeX_Submission.zip (~13 MB)
   - Complete LaTeX source with all dependencies
   - Upload as "Main Document - LaTeX"

2. B-FedPLC_Manuscript.pdf (~6 MB)
   - Final formatted manuscript (14 pages)
   - Upload as "Main Document - PDF"

3. Conflict_of_Interest.txt
   - Statement of no conflicts
   - Upload in COI section

4. Cover_Letter.txt
   - Cover letter to Editor
   - Upload as cover letter

=================================================================
SUBMISSION WEBSITE
=================================================================

https://mc.manuscriptcentral.com/ieee-access

=================================================================
MANUSCRIPT DETAILS
=================================================================

Title: B-FedPLC: Blockchain-Enabled Federated Learning with 
       Prototype-Anchored Learning and Dynamic Community Adaptation 
       for Byzantine-Resilient Distributed Machine Learning

Type: Research Article

Keywords (10):
  1. Audit trail
  2. Blockchain
  3. Byzantine fault tolerance
  4. Dynamic clustering
  5. Federated learning
  6. IPFS
  7. Merkle tree verification
  8. Non-IID data
  9. Personalized learning
  10. Prototype learning

Authors (6):
  1. Rachmad Andri Atmoko (ORCID: 0000-0001-9787-4625)
  2. Sholeh Hadi Pramono* (ORCID: 0000-0003-4399-284X) *Corresponding
  3. Muhammad Fauzan Edy Purnomo (ORCID: 0000-0001-8212-9366)
  4. Panca Mudjirahardjo (ORCID: 0000-0001-5097-2658)
  5. Mahdin Rohmatillah (ORCID: 0000-0001-8417-2165)
  6. Cries Avian (ORCID: 0000-0002-4968-3450)

Affiliation: Universitas Brawijaya, Indonesia

Statistics:
  - Pages: 14
  - Figures: 10
  - Tables: 9
  - Algorithms: 2
  - References: 61
  - Total Package Size: ~19 MB

=================================================================
SEE SUBMISSION_INSTRUCTIONS.md FOR DETAILED STEP-BY-STEP GUIDE
=================================================================
"""

readme_path = os.path.join(dest_dir, 'README.txt')
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("✓ Created: README.txt")

# 5. Summary
print("\n[5/5] Creating summary...")
print("\n" + "="*70)
print("✅ SUBMISSION PACKAGE COMPLETE!")
print("="*70)
print(f"\nLocation: {dest_dir}")
print("\nPackage Contents:")
files = os.listdir(dest_dir)
for i, f in enumerate(sorted(files), 1):
    size = os.path.getsize(os.path.join(dest_dir, f))
    if size > 1024*1024:
        size_str = f"{size/(1024*1024):.2f} MB"
    elif size > 1024:
        size_str = f"{size/1024:.2f} KB"
    else:
        size_str = f"{size} bytes"
    print(f"  {i}. {f:<45} {size_str:>12}")

total_size = sum(os.path.getsize(os.path.join(dest_dir, f)) for f in files) / (1024*1024)
print(f"\nTotal Size: {total_size:.2f} MB (within IEEE 40 MB limit)")
print("\n🚀 READY TO SUBMIT TO IEEE ACCESS!")
print("="*70)
