from pathlib import Path
import SimpleITK as sitk

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "src" / "data" / "raw"
PICAI_DIR = RAW_DATA_DIR / "picai"
LABELS_DIR = RAW_DATA_DIR / "picai_labels"


def listar_arquivos_mha():
    arquivos = list(PICAI_DIR.rglob("*.mha"))

    print(f"Total de arquivos .mha encontrados: {len(arquivos)}")

    for arquivo in arquivos[:10]:
        print(arquivo)


def carregar_mha(caminho):
    imagem = sitk.ReadImage(str(caminho))
    array = sitk.GetArrayFromImage(imagem)

    print("Arquivo:", caminho.name)
    print("Shape:", array.shape)
    print("Spacing:", imagem.GetSpacing())
    print("Origin:", imagem.GetOrigin())
    print("Direction:", imagem.GetDirection())

    return imagem, array

def carregar_nifti(caminho):
    imagem = sitk.ReadImage(str(caminho))
    array = sitk.GetArrayFromImage(imagem)

    print("Arquivo:", caminho.name)
    print("Shape:", array.shape)
    print("Spacing:", imagem.GetSpacing())
    print("Origin:", imagem.GetOrigin())
    print("Direction:", imagem.GetDirection())

    return imagem, array

if __name__ == "__main__":
    t2w_path = PICAI_DIR / "10000" / "10000_1000000_t2w.mha"

    mask_path = (
        LABELS_DIR
        / "anatomical_delineations"
        / "whole_gland"
        / "AI"
        / "Bosma22b"
        / "10000_1000000.nii.gz"
    )

    print("\n--- T2W ---")
    t2w_img, t2w_array = carregar_mha(t2w_path)

    print("\n--- MASK ---")
    mask_img, mask_array = carregar_nifti(mask_path)