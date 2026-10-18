# PI-CAI — Segmentação, Radiômica e Classificação de csPCa

Projeto de Trabalho de Conclusão de Curso que investiga uma pipeline de segmentação da próstata, extração de características radiômicas e classificação da presença de câncer de próstata clinicamente significativo (**csPCa**) por exame.

A implementação atual utiliza imagens **T2W** do PI-CAI, uma **U-Net 2D** em PyTorch, **PyRadiomics** e os classificadores **Regressão Logística, Random Forest e XGBoost**. A segmentação é anatômica, da glândula inteira; a classificação usa o desfecho `case_csPCa`, sem localizar lesões. Embora o dataset inclua outras sequências, a pipeline atual utiliza T2W.

## Estado atual

- Preparação dos dados, treinamento, avaliação e inferência da U-Net implementados.
- Máscaras preditas reconstruídas em volumes 3D e utilizadas na extração radiômica.
- Tabelas radiômicas disponíveis para glândula inteira predita, zonas periférica e de transição, glândula inteira de referência e união das zonas.
- Comparações entre classificadores e representações anatômicas disponíveis em notebooks e em `results/`.
- Reconstrução do experimento histórico de XGBoost preservada separadamente.
- Divisão alternativa agrupada por paciente disponível; a migração e a reavaliação completa da pipeline ainda precisam ser consolidadas.

### Situação da validação

O arquivo original `src/data/processed/splits/patient_splits.csv` contém **1.500 exames de 1.476 pacientes**, distribuídos em **1.050 / 225 / 225 exames** para treino, validação e teste. A divisão original foi estratificada por `case_csPCa`, com semente 42, mas **10 pacientes aparecem em mais de uma partição**.

Os notebooks principais ainda apontam para essa divisão original. Os resultados associados a ela devem ser tratados como históricos e exploratórios, com risco de vazamento entre pacientes.

O arquivo `patient_splits_grouped_v2.csv`, no mesmo diretório, contém **1.049 / 226 / 225 exames**, com rótulos `train`, `val` e `test`, e **nenhum paciente compartilhado entre partições**. Sua utilização exige adequar os consumidores que esperam o rótulo `validation`, além de retreinar e reavaliar os modelos. A existência desse arquivo não indica que os resultados já tenham sido recalculados.

## Pipeline implementada

```text
Volumes T2W + máscaras anatômicas + informações clínicas
                         │
                         ▼
Validação de geometria e preparação de slices 2D (256 × 256)
                         │
                         ▼
U-Net 2D → inferência → máscara predita 3D da glândula inteira
                         │
                         ▼
PyRadiomics no volume T2W original e na máscara 3D
                         │
                         ▼
Regressão Logística / Random Forest / XGBoost
                         │
                         ▼
Seleção na validação → avaliação de csPCa por exame
```

As análises adicionais utilizam máscaras anatômicas de referência de **PZ** (zona periférica), **TZ** (zona de transição), glândula inteira e união PZ/TZ. **PZ + TZ** é a concatenação das características das duas zonas; **união PZ/TZ** é a extração sobre uma única máscara formada pela união.

## Estrutura do projeto

```text
TCC_II/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_segmentation_training.ipynb
│   ├── 03_segmentation_evaluatin.ipynb
│   ├── 04_segmentation_inference.ipynb
│   ├── 05_radiomics.ipynb
│   ├── 06_xgboost_classification_nao_rodar.ipynb
│   ├── 06_xgboost_classification_reconstrucao_original.ipynb
│   ├── 07_baseline_comparison.ipynb
│   ├── 08_zonal_radiomics.ipynb
│   ├── 09_zonal_classification.ipynb
│   ├── 10_reference_radiomics.ipynb
│   ├── 11_reference_comparison.ipynb
│   ├── test.ipynb
├── src/
│   ├── data/
│   │   ├── paths.py
│   │   ├── loader.py
│   │   ├── mapping.py
│   │   ├── splits.py
│   │   ├── raw/
│   │   │   ├── picai/
│   │   │   ├── picai_labels/
│   │   ├── processed/
│   │   │   ├── whole_gland_2d/
│   │   │   │   ├── images/
│   │   │   │   ├── masks/
│   │   │   ├── splits/
│   │   ├── features/
│   ├── preprocessing/
│   │   ├── preprocessing.py
│   │   ├── slicing.py
│   │   ├── dataset_builder.py
│   ├── segmentation/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── train.py
│   │   ├── inference.py
│   ├── visualization/
│       ├── visualization.py
├── results/
│   ├── checkpoints/
│   ├── segmentation/
│   │   ├── predicted_masks/
│   ├── classification/
│   ├── classification_original_reconstruction/
│   ├── baseline_comparison/
│   ├── zonal_radiomics/
│   ├── zonal_classification/
│   ├── reference_radiomics/
│   ├── reference_comparison/
├── README.md
```

Dados brutos, intermediários, checkpoints e máscaras são recursos locais ou gerados pela execução; não se deve presumir que estejam disponíveis em um clone. A lógica de radiômica e classificação está nos notebooks; os módulos reutilizáveis estão em `src/`.

## Dados e pré-processamento

Os caminhos centrais estão definidos em [src/data/paths.py](src/data/paths.py):

- Imagens: `src/data/raw/picai/`.
- Labels: `src/data/raw/picai_labels/`.
- Máscaras da glândula inteira: `anatomical_delineations/whole_gland/AI/`, dentro dos labels.
- Informações clínicas: `clinical_information/marksheet.csv`, dentro dos labels.
- Partições: `src/data/processed/splits/`.

A associação utiliza `patient_id` e `study_id`. Antes do processamento, são verificadas dimensões e propriedades espaciais, incluindo espaçamento, origem e direção.

O pré-processamento seleciona slices com próstata e duas fatias de contexto antes e depois da região, aplica normalização baseada nos pixels positivos e redimensiona para **256 × 256**. Imagens usam interpolação linear; máscaras usam vizinho mais próximo. O notebook de preparação registra **28.110 pares imagem–máscara**, armazenados como arrays NumPy.

## Segmentação e radiômica

A U-Net recebe uma imagem T2W de um canal e produz logits para segmentação binária. O treinamento combina **BCE e Dice Loss**, com otimizador Adam. O checkpoint histórico é salvo em `results/checkpoints/unet_whole_gland_best.pth`. A avaliação inclui perda, Dice, Dice em slices positivos e IoU.

A extração radiômica usa o volume T2W original e a máscara 3D:

| Parâmetro | Valor |
| --- | --- |
| Tipo de imagem | Original |
| Normalização | `normalize=True`, `normalizeScale=100` |
| Discretização | `binWidth=5` |
| Espaçamento de reamostragem | `[0.5, 0.5, 3.0]` mm |
| Interpolação da imagem | B-spline |
| Classes | firstorder, shape, GLCM, GLRLM, GLSZM, GLDM e NGTDM |

São extraídas **107 características por região**. As tabelas locais atuais possuem 1.500 linhas cada:

| Arquivo em `src/data/features/` | Conteúdo |
| --- | --- |
| `whole_gland_t2w_radiomics.csv` | Identificadores + 107 características da máscara predita |
| `zonal_t2w_radiomics.csv` | Identificadores + 107 características de PZ + 107 de TZ |
| `reference_whole_gland_t2w_radiomics.csv` | Identificadores + 107 da glândula de referência + 107 da união PZ/TZ |

As extrações permitem retomada por CSV e registram falhas. Nos experimentos zonais, confirme a fonte das anotações e os valores dos rótulos antes da execução. As máscaras são alinhadas à geometria T2W em coordenadas físicas; reamostragem não substitui registro ou inspeção visual.

## Execução dos notebooks

Use um kernel com diretório de trabalho em `notebooks/`: vários arquivos calculam a raiz com `Path.cwd().parent`.

| Notebook | Função |
| --- | --- |
| [01](notebooks/01_data_preparation.ipynb) | Inspeção, preparação dos slices e gravação da divisão original |
| [02](notebooks/02_segmentation_training.ipynb) | Treinamento da U-Net com os dados processados |
| [03](notebooks/03_segmentation_evaluatin.ipynb) | Avaliação do checkpoint; nome do arquivo preservado como está no projeto |
| [04](notebooks/04_segmentation_inference.ipynb) | Inferência e geração das máscaras 3D |
| [05](notebooks/05_radiomics.ipynb) | Extração radiômica com máscaras preditas |
| [06 — reconstrução](notebooks/06_xgboost_classification_reconstrucao_original.ipynb) | Reconstrução histórica do XGBoost, com saídas separadas |
| [07](notebooks/07_baseline_comparison.ipynb) | Regressão Logística e Random Forest; comparação com XGBoost arquivado |
| [08](notebooks/08_zonal_radiomics.ipynb) | Extração de PZ e TZ usando máscaras de referência |
| [09](notebooks/09_zonal_classification.ipynb) | Comparação de PZ, TZ, PZ + TZ e glândula predita na mesma coorte |
| [10](notebooks/10_reference_radiomics.ipynb) | Extração da glândula de referência e união PZ/TZ; avaliação da sobreposição das máscaras |
| [11](notebooks/11_reference_comparison.ipynb) | Comparação das seis representações na interseção dos casos disponíveis |

`06_xgboost_classification_nao_rodar.ipynb` está preservado como histórico e não integra a sequência recomendada. A reconstrução original não recupera o histórico completo da busca de hiperparâmetros e não substitui uma nova avaliação com a divisão agrupada.

Antes de uma nova rodada experimental, configure consistentemente as partições, os checkpoints e os diretórios de saída em todas as etapas. Executar a sequência atual sem adaptação reutiliza a divisão original e pode sobrescrever artefatos históricos.

## Ambiente

O projeto utiliza Python, Jupyter, NumPy, pandas, matplotlib, SimpleITK, OpenCV, PyTorch, scikit-learn, XGBoost, joblib e PyRadiomics. CUDA é utilizada pela segmentação quando disponível.

Há dois ambientes locais: `.venv` e `.venv-radiomics`. O notebook 05 documenta um ambiente dedicado de **Python 3.8** para radiômica; os notebooks 08 e 10 seguem esse ambiente. Os demais utilizam o ambiente de treinamento/classificação. Selecione o kernel correspondente antes de executar cada notebook.

O repositório ainda não contém `requirements.txt`, `pyproject.toml` ou um arquivo de ambiente com versões fixadas. A instalação reproduzível das dependências permanece pendente; os ambientes locais não são fornecidos pelo Git.

## Resultados e interpretação

Os artefatos incluem métricas CSV, predições por exame, configurações JSON, modelos e figuras ROC, Precision–Recall e matrizes de confusão. Os principais pontos de consulta são:

- [Métricas históricas de segmentação](results/segmentation/unet_whole_gland_metrics.csv).
- [Comparação de baselines](results/baseline_comparison/test_metrics.csv).
- [Comparação zonal](results/zonal_classification/test_metrics.csv).
- [Comparação com máscaras de referência](results/reference_comparison/test_metrics.csv).
- [Qualidade da união PZ/TZ versus glândula inteira](results/reference_radiomics/union_vs_whole_gland_quality.csv).

Nos notebooks comparativos, os candidatos são selecionados por ROC-AUC de validação e os limiares pelo índice de Youden na validação. As métricas de teste incluem ROC-AUC, Average Precision (coluna `pr_auc`), sensibilidade, especificidade, precisão, F1, acurácia e acurácia balanceada. Imputação e padronização, quando aplicáveis, são ajustadas no treino.

As comparações exigem os mesmos casos e partições. As máscaras zonais são anotações anatômicas de referência, enquanto a pipeline original utiliza máscaras preditas: diferenças de desempenho não podem ser atribuídas apenas à região. A união PZ/TZ também não é automaticamente equivalente à glândula inteira.

Além da sobreposição de pacientes na divisão original, o conjunto de teste já foi inspecionado em análises anteriores. Os experimentos adicionais são exploratórios e não constituem validação externa ou confirmação independente.

## Próximos passos

- Consolidar o uso da divisão agrupada por paciente em toda a pipeline.
- Retreinar e reavaliar a segmentação, regenerar as máscaras e recalcular os experimentos dependentes em diretórios separados.
- Consolidar a avaliação volumétrica 3D e seus relatórios de falhas.
- Registrar versões das dependências e configurações para reprodução.
- Consolidar tabelas e figuras do TCC, distinguindo resultados históricos dos reavaliados.

Projeto de finalidade acadêmica e experimental, sem validação para uso clínico.
