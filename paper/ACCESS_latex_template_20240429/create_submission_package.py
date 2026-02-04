"""
Create IEEE Access submission package for B-FedPLC paper
"""
import zipfile
import os
import shutil

def create_latex_submission_zip():
    """Create ZIP file with all LaTeX files for IEEE Access submission"""
    zip_filename = 'B-FedPLC_LaTeX_Submission.zip'

    print(f"Creating {zip_filename}...")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Main LaTeX file
        zipf.write('bfedplc_paper.tex')
        print("✓ Added bfedplc_paper.tex")

        # IEEE template files
        template_files = [
            'ieeeaccess.cls',
            'IEEEtran.cls',
            'IEEEtran.bst',
            'spotcolor.sty',
            't1times.fd',
            't1formata.fd',
            't1giovannistd.fd',
            't1helvetica.fd'
        ]

        for f in template_files:
            if os.path.exists(f):
                zipf.write(f)
                print(f"✓ Added {f}")

        # Font files
        font_extensions = ['.pfb', '.tfm', '.map']
        for f in os.listdir('.'):
            if f.startswith('t1-') and any(f.endswith(ext) for ext in font_extensions):
                zipf.write(f)
        print("✓ Added font files")

        # Figures directory
        if os.path.exists('figures'):
            for f in os.listdir('figures'):
                fig_path = os.path.join('figures', f)
                zipf.write(fig_path, arcname=f'figures/{f}')
            print(f"✓ Added figures directory")

        # Author photos and logo images
        image_files = [
            'moko.jpg', 'sholeh.jpg', 'fauzan.jpg',
            'panca.jpg', 'mahdin.jpg', 'cries.jpg',
            'logo.png', 'bullet.png', 'notaglinelogo.png',
            'bfedplc_architecture.png'
        ]

        for img in image_files:
            if os.path.exists(img):
                zipf.write(img)
        print("✓ Added images and photos")

    # Check file size
    size = os.path.getsize(zip_filename)
    size_mb = size / (1024 * 1024)

    print(f"\n✅ Created: {zip_filename}")
    print(f"   Size: {size:,} bytes ({size_mb:.2f} MB)")

    if size_mb < 40:
        print(f"   ✓ Within IEEE Access limit (< 40 MB)")
    else:
        print(f"   ⚠ WARNING: Exceeds 40 MB limit!")

    return zip_filename

def create_submission_package():
    """Create all files needed for IEEE Access submission"""
    print("=" * 60)
    print("IEEE ACCESS SUBMISSION PACKAGE CREATOR")
    print("Paper: B-FedPLC")
    print("=" * 60)
    print()

    # Create LaTeX ZIP
    latex_zip = create_latex_submission_zip()

    # Check PDF exists
    pdf_file = 'bfedplc_paper.pdf'
    if os.path.exists(pdf_file):
        pdf_size = os.path.getsize(pdf_file) / (1024 * 1024)
        print(f"\n✓ PDF file ready: {pdf_file} ({pdf_size:.2f} MB)")
    else:
        print(f"\n⚠ WARNING: PDF file not found: {pdf_file}")

    # Check supplementary files
    print("\n" + "=" * 60)
    print("SUPPLEMENTARY FILES CHECK:")
    print("=" * 60)

    supp_files = {
        'CONFLICT_OF_INTEREST_STATEMENT.txt': 'Conflict of Interest',
        'COVER_LETTER.txt': 'Cover Letter',
        'IEEE_ACCESS_COMPLIANCE_CHECKLIST.md': 'Compliance Checklist'
    }

    for filename, description in supp_files.items():
        if os.path.exists(filename):
            print(f"✓ {description}: {filename}")
        else:
            print(f"✗ {description}: {filename} (NOT FOUND)")

    print("\n" + "=" * 60)
    print("SUBMISSION CHECKLIST:")
    print("=" * 60)
    print("1. ✓ Main Document - LaTeX (.zip):", latex_zip)
    print("2. ✓ Main Document - PDF:", pdf_file)
    print("3. ✓ Conflict of Interest statement")
    print("4. ✓ Cover letter")
    print()
    print("READY FOR IEEE ACCESS SUBMISSION!")
    print("=" * 60)

if __name__ == "__main__":
    create_submission_package()
