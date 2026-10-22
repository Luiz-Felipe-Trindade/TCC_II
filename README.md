# PI-CAI — Segmentação, Radiômica e Classificação de csPCa

Projeto de Trabalho de Conclusão de Curso que investiga a segmentação da próstata, a extração de características radiômicas e a classificação da presença de câncer de próstata clinicamente significativo (**csPCa**) por exame.

A pipeline utiliza imagens **T2W** do PI-CAI, uma **U-Net 2D** em PyTorch, **PyRadiomics** e os classificadores **Regressão Logística, Random Forest e XGBoost**. A segmentação é anatômica, da glândula inteira; a classificação utiliza `case_csPCa`, sem localizar lesões. As demais sequências do dataset não são utilizadas na pipeline atual.

## Organização dos experimentos

A sequência atual está em `notebooks/*_grouped_v2.ipynb` e utiliza `patient_splits_grouped_v2.csv`. Os notebooks históricos estão em `notebooks/archived_do_not_run/` e não devem ser executados como parte da sequência atual.

| Divisão | Treino | Validação | Teste | Sobreposição de pacientes |
| --- | ---: | ---: | ---: | --- |
| Original: `patient_splits.csv` | 1.050 | 225 | 225 | 10 pacientes em mais de uma partição |
| Agrupada: `patient_splits_grouped_v2.csv` | 1.049 | 226 | 225 | Nenhuma |

A coorte contém **1.500 exames de 1.476 pacientes**. A divisão agrupada utiliza `train`, `val` e `test`. O notebook 01 valida o manifesto existente, a coorte e a distribuição das classes; não gera nem sobrescreve esse arquivo. A reprodução exata depende de restaurar o manifesto utilizado no experimento.

Há artefatos locais e saídas de execução dos experimentos agrupados. A presença de um arquivo, isoladamente, não comprova sua procedência: checkpoint, máscaras, características e resultados devem pertencer à mesma rodada. Os resultados originais permanecem históricos, com risco de vazamento entre pacientes. Como o conjunto de teste já foi inspecionado anteriormente, as análises adicionais continuam exploratórias, mesmo com a divisão agrupada.

## Pipeline

```text
Volumes T2W + máscaras anatômicas + informações clínicas
                         ↓
Validação de geometria e preparação de slices 2D (256 × 256)
                         ↓
U-Net 2D → inferência → máscara predita 3D da glândula inteira
                         ↓
PyRadiomics no volume T2W original e na máscara 3D
                         ↓
Regressão Logística / Random Forest / XGBoost
                         ↓
Seleção na validação → avaliação de csPCa por exame
```

As análises adicionais usam máscaras anatômicas de referência de **PZ** (zona periférica), **TZ** (zona de transição), glândula inteira e união PZ/TZ. **PZ + TZ** concatena as características das duas zonas; **união PZ/TZ** extrai características de uma única máscara formada pela união.

## Estrutura do projeto

```text
TCC_II/
├── notebooks/
│   ├── 01_data_preparation_grouped_v2.ipynb
│   ├── ...
│   ├── 11_reference_comparison_grouped_v2.ipynb
│   ├── archived_do_not_run/
├── src/
│   ├── data/
│   │   ├── paths.py
│   │   ├── loader.py
│   │   ├── mapping.py
│   │   ├── splits.py
│   │   ├── raw/                   # Dataset local, ignorado pelo Git
│   │   ├── processed/             # Slices, partições e predições locais
│   │   ├── features/              # Tabelas radiômicas
│   ├── preprocessing/
│   ├── segmentation/
│   ├── visualization/
├── results/                       # Métricas, modelos, relatórios e figuras
├── requirements.txt
├── requirements_radiomics.txt
├── README.md
```

Os caminhos centrais estão em [src/data/paths.py](src/data/paths.py). A lógica de radiômica e classificação está nos notebooks; os módulos reutilizáveis estão em `src/`.

## Preparação do dataset após clonar

Os diretórios `src/data/raw/` e `src/data/processed/` são ignorados integralmente pelo [.gitignore](.gitignore). Portanto, um clone não fornece as imagens, as anotações, os slices ou os manifestos de partição. Os checkpoints e determinados diretórios de máscaras em `results/` também são ignorados.

### 1. Crie os diretórios locais

Na raiz do repositório, execute no PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path src/data/raw/picai, src/data/raw/picai_labels, src/data/processed/splits
```

### 2. Adicione as imagens do PI-CAI

Obtenha os volumes do dataset PI-CAI e extraia as pastas dos pacientes diretamente em `src/data/raw/picai/`. Se as imagens estiverem distribuídas em vários pacotes, reúna as pastas de pacientes nesse mesmo diretório, preservando os arquivos de todos os exames.

O código espera **uma pasta por paciente**, com os volumes diretamente dentro dela:

```text
src/data/raw/picai/
├── <patient_id>/
    └── <patient_id>_<study_id>_t2w.mha
    └── <patient_id>_<study_id>_adc.mha   # Se disponível; não usado atualmente
    └── <patient_id>_<study_id>_hbv.mha   # Se disponível; não usado atualmente
```

Exemplo de caminho esperado: `src/data/raw/picai/10000/10000_1000000_t2w.mha`. Preserve os identificadores e os nomes dos arquivos. Evite uma pasta adicional do pacote entre `picai/` e as pastas dos pacientes: a descoberta de imagens não busca recursivamente em níveis arbitrários.

### 3. Adicione as anotações e as informações clínicas

Copie o conteúdo do pacote de labels para `src/data/raw/picai_labels/`, mantendo a estrutura interna. `clinical_information/` e `anatomical_delineations/` devem ficar diretamente dentro de `picai_labels/`, sem uma pasta intermediária como `picai_labels/picai_labels/`.

```text
src/data/raw/picai_labels/
├── clinical_information/
│   ├── marksheet.csv
├── anatomical_delineations/
│   ├── whole_gland/
│   │   ├── AI/
│   │       └── <fonte>/
│   │           └── <patient_id>_<study_id>.nii.gz
│   ├── zonal_pz_tz/
│       └── AI/
│           └── HeviAI23/
│               └── <patient_id>_<study_id>.nii.gz
├── csPCa_lesion_delineations/         # Se incluído no pacote; não usado na pipeline atual
```

As fontes locais da glândula inteira incluem `Bosma22b` e `Guerbet23`. A busca dessas máscaras é recursiva dentro de `whole_gland/AI/`. Os notebooks zonais selecionam `HeviAI23`; confira a fonte e a correspondência dos rótulos configurados (`PZ_LABEL=1`, `TZ_LABEL=2`) antes de extrair características.

**Atenção à capitalização:** o diretório local é `zonal_pz_tz`, mas há notebooks que usam `Zonal_PZ_TZ`. Em sistemas que diferenciam maiúsculas de minúsculas, ajuste o caminho nos notebooks 08 e 10 para corresponder ao diretório extraído.

Para a segmentação e a radiômica da glândula inteira, são necessários T2W, máscaras da glândula e `marksheet.csv`. As máscaras zonais são necessárias para as etapas 08–11. As máscaras de lesões não são a entrada da segmentação anatômica deste projeto.

### 4. Restaure a divisão agrupada do experimento

Copie o manifesto da cópia local ou do backup do experimento para:

```text
src/data/processed/splits/patient_splits_grouped_v2.csv
```

O CSV deve conter `patient_id`, `study_id` e `split`, com 1.500 pares únicos, partições `train`/`val`/`test` e contagens **1.049 / 226 / 225**. Cada paciente deve aparecer em uma única partição. O notebook 01 também confere os casos contra o `marksheet.csv` e a distribuição de `case_csPCa`.

Esse manifesto não acompanha o clone porque está em um diretório ignorado. Sem uma cópia dele, a sequência agrupada não pode reproduzir exatamente o experimento. A função histórica em `src/data/splits.py` não substitui esse manifesto: ela divide linhas de exames e pode compartilhar pacientes entre partições. A geração original da divisão agrupada, incluindo algoritmo e semente, ainda precisa ser documentada.

Se precisar consultar o experimento histórico, restaure `patient_splits.csv` no mesmo diretório, mantendo os dois arquivos separados.

### 5. Gere ou restaure os arquivos intermediários

| Recurso | Destino | Como obter |
| --- | --- | --- |
| Slices de imagens | `src/data/processed/whole_gland_2d/images/` | Gerar pela etapa de processamento do notebook 01 ou restaurar backup compatível |
| Slices de máscaras | `src/data/processed/whole_gland_2d/masks/` | Gerar junto com as imagens no notebook 01 |
| Checkpoint agrupado | `results/checkpoints/unet_whole_gland_grouped_v2_best.pth` | Treinar no notebook 02 ou restaurar o checkpoint correspondente |
| Máscaras 3D agrupadas | `src/data/processed/whole_gland_predictions_grouped_v2/` | Restaurar ou gerar por inferência com o checkpoint agrupado |

As máscaras esperadas pelo notebook 05 usam o nome `<patient_id>_<study_id>_whole_gland_pred.mha`, cobrindo todos os 1.500 exames. O notebook 04 contém verificações e salva uma predição de inspeção em `results/segmentation/predicted_masks_grouped_v2_check/`; essa saída isolada não substitui o conjunto completo. Confira a configuração da inferência em lote antes de iniciar a extração radiômica.

O notebook 01 apresenta a etapa de processamento dos slices; confira se a chamada de geração está habilitada ao preparar um clone novo. Restaurar intermediários só é adequado quando sua configuração e procedência correspondem ao experimento desejado.

### 6. Confira a instalação dos dados

No PowerShell, a partir da raiz:

```powershell
Test-Path src/data/raw/picai
Test-Path src/data/raw/picai_labels/clinical_information/marksheet.csv
Test-Path src/data/raw/picai_labels/anatomical_delineations/whole_gland/AI
Test-Path src/data/processed/splits/patient_splits_grouped_v2.csv
(Get-ChildItem src/data/raw/picai -Recurse -Filter *_t2w.mha).Count
```

Os testes de caminho devem retornar `True`; para a coorte completa, a contagem esperada é 1.500 volumes T2W. Execute as verificações do notebook 01 para confirmar associação entre imagens, máscaras, informações clínicas e partições, além da geometria dos volumes. A contagem de arquivos não substitui essas verificações.

Os dados devem permanecer nos diretórios ignorados; não é necessário alterar o `.gitignore` ou forçar sua inclusão no Git. As tabelas em `src/data/features/` não estão cobertas pelas regras de exclusão de dados brutos e processados.

## Ambiente

Há dois ambientes separados:

| Ambiente | Python local de referência | Dependências | Uso |
| --- | --- | --- | --- |
| `.venv` | 3.13 | [requirements.txt](requirements.txt) | Preparação, segmentação e classificação |
| `.venv-radiomics` | 3.8 | [requirements_radiomics.txt](requirements_radiomics.txt) | Extração radiômica: notebooks 05, 08 e 10 |

Exemplo de criação e registro dos kernels no Windows, com essas versões do Python instaladas:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name tcc-ii --display-name "TCC II"

py -3.8 -m venv .venv-radiomics
.\.venv-radiomics\Scripts\python.exe -m pip install -r requirements_radiomics.txt
.\.venv-radiomics\Scripts\python.exe -m ipykernel install --user --name tcc-ii-radiomics --display-name "TCC II Radiomics"
```

Os arquivos registram versões dos ambientes locais e incluem dependências específicas de Windows e builds CUDA do PyTorch. A instalação em outra plataforma pode exigir adaptação; a reprodução por instalação limpa ainda precisa ser validada. CUDA é utilizada na segmentação quando disponível. Os ambientes virtuais não acompanham o Git.

## Execução dos notebooks

Abra o Jupyter a partir de `notebooks/` e selecione o kernel correspondente:

```powershell
cd notebooks
..\.venv\Scripts\python.exe -m jupyter lab
```

Vários notebooks calculam a raiz com `Path.cwd().parent`; confirme que o diretório de trabalho do kernel é `notebooks/`.

| Etapa | Notebook atual | Função |
| --- | --- | --- |
| 01 | [Preparação](notebooks/01_data_preparation_grouped_v2.ipynb) | Inspeção, preparação dos slices e validação do manifesto agrupado |
| 02 | [Treinamento](notebooks/02_segmentation_training_grouped_v2.ipynb) | Treinamento da U-Net |
| 03 | [Avaliação](notebooks/03_segmentation_evaluatin_grouped_v2.ipynb) | Avaliação do checkpoint; grafia do nome preservada |
| 04 | [Inferência](notebooks/04_segmentation_inference_grouped_v2.ipynb) | Inferência, inspeção de máscaras e avaliação volumétrica |
| 05 | [Radiômica](notebooks/05_radiomics_grouped_v2.ipynb) | Extração com máscaras preditas |
| 06 | [XGBoost](notebooks/06_xgboost_classification_grouped_v2.ipynb) | Classificação com XGBoost |
| 07 | [Baselines](notebooks/07_baseline_comparison_grouped_v2.ipynb) | Regressão Logística, Random Forest e comparação com XGBoost |
| 08 | [Radiômica zonal](notebooks/08_zonal_radiomics_grouped_v2.ipynb) | Extração de PZ e TZ |
| 09 | [Classificação zonal](notebooks/09_zonal_classification_grouped_v2.ipynb) | Comparação de PZ, TZ, PZ + TZ e glândula predita |
| 10 | [Radiômica de referência](notebooks/10_reference_radiomics_grouped_v2.ipynb) | Glândula de referência, união PZ/TZ e qualidade da sobreposição |
| 11 | [Comparação de referência](notebooks/11_reference_comparison_grouped_v2.ipynb) | Comparação das seis representações na mesma coorte |

Antes de reexecutar, confira as entradas, o checkpoint e os diretórios de saída. As extrações permitem retomada por CSV; reutilizar uma tabela existente com configuração diferente pode misturar experimentos. Preserve os resultados históricos e use uma saída separada para uma nova rodada.

## Passo a passo para executar toda a pipeline

Este roteiro considera uma execução completa da versão `grouped_v2`, incluindo as análises zonais e de referência. Use os links da tabela acima para abrir os notebooks. Em cada notebook, selecione o kernel indicado e execute as células na ordem, de cima para baixo. Interrompa a sequência se houver erro ou uma verificação da coorte falhar; corrija a causa antes de continuar.

### 1. Prepare os dados e os ambientes

Conclua as seções **Preparação do dataset após clonar** e **Ambiente**: posicione imagens, anotações e `marksheet.csv`, restaure `patient_splits_grouped_v2.csv`, instale as dependências e registre os dois kernels. Para executar também as etapas zonais, disponibilize as máscaras da fonte selecionada nos notebooks 08 e 10.

Antes de iniciar uma nova rodada, preserve os artefatos existentes. Confira os caminhos de saída em todos os notebooks: checkpoints, máscaras e CSVs existentes podem ser sobrescritos ou reutilizados. Se alterar uma configuração, utilize diretórios e nomes próprios para a nova rodada e atualize as etapas consumidoras de forma consistente.

Abra o Jupyter com os comandos da seção anterior. No primeiro notebook, confirme o diretório do kernel:

```python
from pathlib import Path
assert Path.cwd().name == "notebooks", Path.cwd()
assert (Path.cwd().parent / "src").is_dir()
```

### 2. Prepare os slices — notebook 01

**Kernel: TCC II.** Execute a descoberta dos exames, a inspeção das máscaras e as verificações de geometria. Execute a célula que chama `process_dataset(...)` para gerar os arrays em `src/data/processed/whole_gland_2d/`, caso ainda não tenha intermediários compatíveis.

Confira a correspondência entre `images/` e `masks/`, os arrays de 256 × 256 e o resumo de processamento. A preparação documentada possui 28.110 pares. Ao final, execute a validação do manifesto agrupado: devem existir 1.500 exames, 1.476 pacientes, contagens 1.049/226/225 e nenhum paciente compartilhado entre partições. Esse notebook exige o manifesto restaurado; ele não o gera.

### 3. Treine a U-Net — notebook 02

**Kernel: TCC II.** Confira os caminhos dos slices e do manifesto, os parâmetros de treinamento e o dispositivo selecionado (`cuda` ou `cpu`). Execute a criação dos datasets e loaders, do modelo e do treinamento. Aguarde sua conclusão e confira o histórico de perda e Dice.

A saída esperada é `results/checkpoints/unet_whole_gland_grouped_v2_best.pth`. Esse checkpoint deve ser o mesmo utilizado nas etapas 03 e 04. Se estiver reproduzindo a partir de um checkpoint previamente treinado e de procedência conhecida, restaure-o nesse caminho e avance para a avaliação.

### 4. Avalie a segmentação 2D — notebook 03

**Kernel: TCC II.** Carregue o checkpoint agrupado e execute a avaliação e as visualizações. Confira perda, Dice, Dice em slices positivos e IoU, além das imagens com sobreposição das máscaras. Verifique os caminhos dos relatórios salvos em `results/segmentation/`.

### 5. Gere todas as máscaras 3D — notebook 04

**Kernel: TCC II.** Execute primeiro as células de configuração, carregamento do checkpoint e criação do modelo. Faça a inspeção de um caso com as células que carregam a T2W e calculam sua predição. As células que leem máscaras já salvas no diretório definitivo dependem da geração em lote abaixo.

Para produzir as máscaras de todos os exames, adicione e execute esta célula **depois de carregar o modelo e antes das células que consomem as máscaras agrupadas existentes ou calculam métricas 3D**:

```python
import pandas as pd
from src.segmentation.inference import predict_and_save_cases

splits_path = (
    PROJECT_ROOT / "src/data/processed/splits/patient_splits_grouped_v2.csv"
)
cases = pd.read_csv(
    splits_path, dtype={"patient_id": str, "study_id": str}
)
assert len(cases) == 1500
assert not cases.duplicated(["patient_id", "study_id"]).any()
assert cases.groupby("patient_id")["split"].nunique().max() == 1

prediction_dir = (
    PROJECT_ROOT / "src/data/processed/whole_gland_predictions_grouped_v2"
)
records = predict_and_save_cases(
    model=model,
    cases=cases[["patient_id", "study_id"]],
    device=device,
    output_dir=prediction_dir,
    threshold=0.5,
    image_size=(256, 256),
)

expected = set(cases["patient_id"] + "_" + cases["study_id"])
available = {
    p.name.removesuffix("_whole_gland_pred.mha")
    for p in prediction_dir.glob("*_whole_gland_pred.mha")
}
assert available == expected, (
    f"Faltantes: {len(expected - available)}; extras: {len(available - expected)}"
)
print("Mascaras disponiveis:", len(available))
```

A função pula arquivos já existentes. Para recalcular com outro checkpoint, use um novo diretório e ajuste também os consumidores; a existência da máscara não valida sua origem. A inferência deve cobrir treino, validação e teste, pois a classificação precisa das características de todos os exames. Isso não envolve ajustar novamente a U-Net.

Após gerar as 1.500 máscaras, execute as células de comparação e a avaliação volumétrica do teste no notebook 04. Confira os 225 exames e investigue as linhas com `status=error` em `results/segmentation/grouped_v2_test_3d_metrics.csv`. O resumo está em `grouped_v2_test_3d_summary.csv`. A máscara de inspeção em `predicted_masks_grouped_v2_check/` é uma saída separada da inferência em lote.

### 6. Extraia a radiômica da glândula predita — notebook 05

**Kernel: TCC II Radiomics.** Confirme o kernel antes de importar PyRadiomics. Execute a conferência das máscaras, a configuração do extrator e a extração em lote. O notebook exige correspondência exata com os 1.500 exames clínicos.

Confira `src/data/features/whole_gland_t2w_radiomics_grouped_v2.csv`: a tabela completa deve conter 1.500 linhas e 107 características por exame, além dos identificadores. Examine também o CSV de falhas. Se a extração for interrompida, a retomada deve utilizar a mesma configuração e as mesmas máscaras.

### 7. Treine e compare os classificadores — notebooks 06 e 07

**Kernel: TCC II.** Execute o notebook 06 para o XGBoost e, em seguida, o 07 para os baselines e a comparação. Confira se ambos usam a tabela agrupada e o mesmo manifesto de partições.

Execute o ajuste no treino e a seleção de candidatos e limiares na validação antes da avaliação final do teste. Examine as métricas, curvas e predições em `results/classification/grouped_v2_xgboost/` e `results/baseline_comparison_grouped_v2/`. A etapa 07 salva, entre outros arquivos, `validation_model_selection.csv`, `test_metrics.csv`, `test_predictions.csv` e `experiment_config.json`.

### 8. Extraia as características zonais — notebook 08

**Kernel: TCC II Radiomics.** Confira o diretório das máscaras zonais, a fonte `HeviAI23`, os rótulos de PZ/TZ e o alinhamento físico com a T2W. Execute as verificações visuais antes da extração completa.

Confira `src/data/features/zonal_t2w_radiomics_grouped_v2.csv`: são esperadas 107 características de PZ e 107 de TZ por exame. Revise cobertura e falhas em `results/zonal_radiomics_grouped_v2/` antes da comparação.

### 9. Compare as representações zonais — notebook 09

**Kernel: TCC II.** Execute a carga das tabelas predita e zonal, a validação da coorte, o treinamento, a seleção na validação e a avaliação do teste. Os resultados comparam PZ, TZ, PZ + TZ e glândula predita em `results/zonal_classification_grouped_v2/`.

### 10. Extraia a radiômica de referência — notebook 10

**Kernel: TCC II Radiomics.** Confira as fontes das máscaras da glândula inteira e das zonas, os caminhos e os rótulos. Execute a inspeção visual, a extração e os relatórios de cobertura e sobreposição.

A tabela de saída é `src/data/features/reference_whole_gland_t2w_radiomics_grouped_v2.csv`, com 107 características da glândula de referência e 107 da união PZ/TZ por exame. Examine os relatórios em `results/reference_radiomics_grouped_v2/`; a união das zonas pode diferir da glândula inteira.

### 11. Compare as seis representações — notebook 11

**Kernel: TCC II.** Execute a carga das três tabelas radiômicas e a validação da coorte compartilhada. O notebook verifica os 1.500 exames, as partições e a ausência de sobreposição de pacientes. Resolva eventuais extrações faltantes antes de prosseguir.

Execute o treinamento e a seleção na validação e, por fim, a avaliação do teste. Consulte as métricas, predições, modelos e figuras em `results/reference_comparison_grouped_v2/`.

### 12. Confira e preserve a rodada

Verifique se os relatórios cobrem os casos esperados e se as falhas foram resolvidas ou explicitamente documentadas. Guarde o manifesto de partições, o checkpoint, as configurações, os relatórios e as tabelas utilizadas. Salve os notebooks com suas saídas e registre a relação entre os artefatos da rodada.

A sequência de kernels é: **TCC II nas etapas 01–04, Radiomics na 05, TCC II nas 06–07, Radiomics na 08, TCC II na 09, Radiomics na 10 e TCC II na 11**. A execução termina com as comparações da glândula predita, das zonas e das máscaras de referência, mantendo a interpretação exploratória descrita neste documento.

## Pré-processamento, segmentação e radiômica

O pré-processamento seleciona slices com próstata e duas fatias de contexto antes e depois, normaliza pelos pixels positivos e redimensiona para **256 × 256**. Imagens usam interpolação linear; máscaras usam vizinho mais próximo. A preparação documenta **28.110 pares imagem–máscara** em arrays NumPy.

A U-Net recebe T2W de um canal e produz logits para segmentação binária. O treinamento combina **BCE e Dice Loss**, com Adam. A avaliação inclui perda, Dice, Dice em slices positivos e IoU; o notebook 04 também contém avaliação 3D com relatório por exame e registro de falhas.

A radiômica utiliza o volume T2W original e máscaras 3D alinhadas em coordenadas físicas:

| Parâmetro | Valor |
| --- | --- |
| Tipo de imagem | Original |
| Normalização | `normalize=True`, `normalizeScale=100` |
| Discretização | `binWidth=5` |
| Espaçamento de reamostragem | `[0.5, 0.5, 3.0]` mm |
| Interpolação da imagem | B-spline |
| Classes | firstorder, shape, GLCM, GLRLM, GLSZM, GLDM e NGTDM |

São extraídas **107 características por região**:

| Arquivo atual em `src/data/features/` | Conteúdo além dos identificadores |
| --- | --- |
| `whole_gland_t2w_radiomics_grouped_v2.csv` | 107 características da glândula predita |
| `zonal_t2w_radiomics_grouped_v2.csv` | 107 de PZ + 107 de TZ |
| `reference_whole_gland_t2w_radiomics_grouped_v2.csv` | 107 da glândula de referência + 107 da união PZ/TZ |

As extrações registram falhas e permitem retomada. Reamostragem não substitui registro ou inspeção visual da correspondência anatômica.

## Resultados e interpretação

Os artefatos incluem métricas CSV, predições por exame, configurações JSON, modelos, curvas ROC e Precision–Recall e matrizes de confusão. Os resultados agrupados são organizados em:

- `results/segmentation/`: avaliações de segmentação; confira o identificador da rodada no nome dos arquivos.
- `results/classification/grouped_v2_xgboost/`: classificação com XGBoost.
- `results/baseline_comparison_grouped_v2/`: comparação de classificadores.
- `results/zonal_radiomics_grouped_v2/` e `results/zonal_classification_grouped_v2/`: análises zonais.
- `results/reference_radiomics_grouped_v2/` e `results/reference_comparison_grouped_v2/`: análises com máscaras de referência.

Nos notebooks comparativos, candidatos são selecionados por ROC-AUC de validação e limiares pelo índice de Youden na validação. As métricas de teste incluem ROC-AUC, Average Precision (coluna `pr_auc`), sensibilidade, especificidade, precisão, F1, acurácia e acurácia balanceada. Imputação e padronização, quando aplicáveis, são ajustadas no treino.

As comparações exigem os mesmos casos e partições. As máscaras zonais são anotações anatômicas de referência, enquanto a pipeline da U-Net utiliza máscaras preditas: diferenças de desempenho não podem ser atribuídas apenas à região. A união PZ/TZ também não é automaticamente equivalente à glândula inteira; interprete os resultados junto aos relatórios de sobreposição.

A ausência de pacientes compartilhados entre partições corrige essa fonte de vazamento, mas não torna o teste anteriormente explorado uma confirmação independente. O projeto não apresenta validação externa.

## Pendências de reprodução

- Documentar a geração original do manifesto agrupado e definir uma forma de distribuí-lo junto aos materiais do experimento.
- Consolidar a procedência dos checkpoints, máscaras, tabelas e resultados de cada rodada.
- Consolidar a geração em lote das máscaras e os relatórios volumétricos de falhas.
- Validar a instalação limpa dos dois ambientes.
- Consolidar tabelas e figuras do TCC, distinguindo resultados históricos e agrupados.

Projeto de finalidade acadêmica e experimental, sem validação para uso clínico.
