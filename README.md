# PI-CAI — Pipeline de Segmentação, Radiômica e Classificação

Projeto de desenvolvimento do Trabalho de Conclusão de Curso voltado à investigação de uma pipeline híbrida para **detecção de câncer de próstata clinicamente significativo (csPCa)** em exames de **Ressonância Magnética Biparamétrica (RMbp)**.

A proposta combina **Deep Learning**, **Radiômica** e **Machine Learning**, utilizando o dataset **PI-CAI** como base para desenvolvimento e avaliação.

---

## 🎯 Objetivo

Desenvolver e avaliar uma pipeline computacional capaz de utilizar imagens de RMbp para:

1. realizar a segmentação das regiões de interesse;
2. extrair características quantitativas por meio de radiômica;
3. selecionar as características mais relevantes;
4. classificar os casos utilizando XGBoost;
5. avaliar o desempenho da abordagem por meio de métricas de segmentação e classificação.

A proposta busca equilibrar **desempenho, interpretabilidade e custo computacional**.

---

## 🧠 Pipeline proposta

```text
                    PI-CAI
                       │
                       ▼
              ┌─────────────────┐
              │ Pré-processamento│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │      U-Net      │
              │   Segmentação   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Máscara     │
              │  Região de ROI  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Radiômica    │
              │  Extração de    │
              │    Features     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Seleção Features│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     XGBoost     │
              │   Classificação │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Avaliação    │
              └─────────────────┘
```

---

## 📊 Dataset

O projeto utiliza o **PI-CAI (Prostate Imaging: Cancer AI)**, um dataset voltado ao desenvolvimento e avaliação de métodos de Inteligência Artificial para detecção de câncer de próstata em imagens de ressonância magnética.

Os dados utilizados no projeto estão organizados em **5 folds**.

### Imagens

Entre as modalidades/sequências presentes estão:

* T2W;
* ADC;
* DWI/HBV;
* demais arquivos associados aos estudos.

Os exames são disponibilizados em formatos como `.mha`.

### Labels

As informações complementares estão organizadas no repositório de labels do PI-CAI e incluem:

* delineações anatômicas;
* delineações de lesões de csPCa;
* informações clínicas;
* recursos adicionais.

Entre as informações clínicas disponíveis estão atributos como:

* `patient_id`;
* `study_id`;
* `mri_date`;
* `patient_age`;
* `psa`;
* `psad`;
* `prostate_volume`;
* `histopath_type`;
* `lesion_PIRADS`;
* `lesion_GS`;
* `lesion_ISUP`;
* `case_ISUP`;
* `case_csPCa`;
* `center`.

> Os dados brutos do PI-CAI e os arquivos de labels não devem ser versionados diretamente no Git deste projeto.

---

## 📁 Estrutura do projeto

```text
TCC/
│
├── data/
│   ├── raw/
│   │   ├── PI-CAI/
│   │   │   ├── fold_0/
│   │   │   ├── fold_1/
│   │   │   ├── fold_2/
│   │   │   ├── fold_3/
│   │   │   └── fold_4/
│   │   │
│   │   └── PI-CAI-labels/
│   │       ├── additional_resources/
│   │       ├── anatomical_delineations/
│   │       ├── clinical_information/
│   │       └── csPCa_lesion_delineations/
│   │
│   ├── processed/
│   │   ├── images/
│   │   ├── masks/
│   │   └── slices/
│   │
│   └── features/
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── dataset.py
│   │
│   ├── preprocessing/
│   │   ├── normalization.py
│   │   └── preprocessing.py
│   │
│   ├── segmentation/
│   │   ├── unet.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── radiomics/
│   │   └── extract.py
│   │
│   ├── classification/
│   │   ├── xgboost_model.py
│   │   └── baselines.py
│   │
│   └── evaluation/
│       ├── segmentation_metrics.py
│       └── classification_metrics.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_visualization.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_unet_experiments.ipynb
│   ├── 05_radiomics.ipynb
│   └── 06_classification.ipynb
│
├── models/
│   ├── unet/
│   └── xgboost/
│
├── results/
│   ├── segmentation/
│   ├── radiomics/
│   └── classification/
│
├── docs/
│
├── README.md
├── ROADMAP.md
├── TODO.md
└── .gitignore
```

---

## 🔬 Etapas do desenvolvimento

### 1. Organização dos dados

Primeiramente serão organizados os arquivos do PI-CAI e seus respectivos labels.

O objetivo é estabelecer uma relação confiável entre:

```text
patient_id
      ↓
study_id
      ↓
imagem
      ↓
máscara
      ↓
informação clínica
```

---

### 2. Exploração e validação

Antes do treinamento, os arquivos serão carregados e visualizados para verificar:

* dimensões;
* orientação;
* espaçamento;
* correspondência entre imagens e máscaras;
* existência de casos incompletos;
* alinhamento espacial.

Essa etapa é importante para evitar que problemas nos dados sejam confundidos posteriormente com problemas do modelo.

---

### 3. Pré-processamento

As imagens serão preparadas para utilização pela U-Net.

A estratégia prevista inclui:

* normalização;
* padronização;
* definição da região de interesse;
* conversão dos volumes 3D em slices 2D;
* associação entre imagem e máscara.

---

### 4. Segmentação

Será investigada uma **U-Net 2D** para segmentação das regiões de interesse.

A avaliação será realizada utilizando principalmente:

* Dice Score;
* Intersection over Union (IoU).

---

### 5. Radiômica

Após a segmentação, serão extraídas características quantitativas das regiões de interesse.

As principais categorias consideradas inicialmente são:

* características de primeira ordem;
* características de forma;
* características de textura.

A extração será realizada utilizando o **PyRadiomics**.

---

### 6. Seleção de features

Como a radiômica pode gerar grande quantidade de atributos, será realizada uma etapa de redução e seleção.

Inicialmente serão investigadas estratégias como:

* análise de correlação;
* remoção de atributos redundantes;
* RFE;
* importância das features pelo XGBoost.

A seleção deverá ser realizada de maneira compatível com a estratégia de validação, evitando data leakage.

---

### 7. Classificação

As características selecionadas serão utilizadas como entrada para o **XGBoost**.

O modelo terá como objetivo classificar os casos quanto à presença de **csPCa**.

Também poderão ser utilizados modelos como:

* Random Forest;
* SVM.

como baselines para comparação.

---

## 📈 Avaliação

### Segmentação

Serão consideradas:

| Métrica    | Objetivo                          |
| ---------- | --------------------------------- |
| Dice Score | Avaliar sobreposição das máscaras |
| IoU        | Avaliar interseção sobre união    |

### Classificação

Serão consideradas:

| Métrica        | Objetivo                            |
| -------------- | ----------------------------------- |
| ROC-AUC        | Avaliar capacidade discriminativa   |
| Sensibilidade  | Avaliar detecção dos positivos      |
| Especificidade | Avaliar identificação dos negativos |
| F1-Score       | Equilibrar precisão e recall        |

Também serão analisados:

* matriz de confusão;
* curva ROC;
* curva Precision-Recall;
* falsos positivos;
* falsos negativos.

---

## 🔒 Validação e prevenção de Data Leakage

A estratégia de validação será definida durante o desenvolvimento experimental.

A princípio, será investigada a utilização de **5-Fold Cross Validation**, considerando a distribuição dos casos.

Cuidados importantes:

* não misturar casos do mesmo paciente entre treino e validação;
* realizar seleção de features somente utilizando os dados de treinamento;
* evitar que informações do conjunto de teste sejam utilizadas durante o treinamento;
* considerar a natureza multicêntrica do PI-CAI.

---

## 🖥️ Ambiente computacional

O desenvolvimento será realizado prioritariamente em ambiente local utilizando:

* **CPU:** Intel Core i7-7700;
* **RAM:** 16 GB;
* **GPU:** NVIDIA RTX 3070.

A GPU será utilizada principalmente para treinamento e experimentação com a U-Net.

O processamento poderá ser dividido entre:

```text
CPU
→ organização dos dados
→ pré-processamento
→ radiômica
→ XGBoost

GPU
→ treinamento da U-Net
→ inferência da U-Net
```

---

## 🛠️ Tecnologias previstas

### Linguagem

* Python

### Deep Learning

* PyTorch
* MONAI

### Processamento de imagens

* SimpleITK
* NiBabel

### Radiômica

* PyRadiomics

### Machine Learning

* XGBoost
* Scikit-learn

### Desenvolvimento

* VS Code
* Git
* GitHub

---

## 📂 Controle de versões

O Git será utilizado para versionar:

* código;
* notebooks;
* configurações;
* documentação;
* scripts;
* resultados selecionados.

Os dados brutos do PI-CAI, modelos grandes e arquivos intermediários não deverão ser armazenados diretamente no repositório.

---

## 📚 Documentação do projeto

A documentação será dividida em três arquivos principais:

### `README.md`

Apresenta:

* objetivo;
* arquitetura;
* dataset;
* tecnologias;
* estrutura do projeto.

### `ROADMAP.md`

Apresenta:

* visão geral;
* etapas;
* dependências;
* evolução planejada.

### `TODO.md`

Apresenta:

* tarefas;
* experimentos;
* itens pendentes;
* progresso do desenvolvimento.

---

## 🚧 Status atual

### Concluído

* [x] Definição da proposta metodológica
* [x] Download/obtenção do repositório de labels
* [x] Organização inicial do projeto
* [x] Identificação da estrutura dos dados
* [x] Definição da pipeline inicial

### Em andamento

* [ ] Download completo dos dados
* [ ] Organização dos 5 folds
* [ ] Implementação do data loader
* [ ] Validação das imagens e máscaras
* [ ] Visualização dos casos

### Próximo marco

> **Carregar um caso completo do PI-CAI e visualizar imagem + máscara corretamente alinhadas.**

Esse será o primeiro teste importante da pipeline.

---

## 📌 Princípio de desenvolvimento

O projeto será desenvolvido de forma incremental.

```text
Organizar
   ↓
Carregar
   ↓
Visualizar
   ↓
Validar
   ↓
Treinar
   ↓
Avaliar
   ↓
Otimizar
```

A prioridade inicial não será obter o melhor resultado possível, mas garantir que **cada etapa esteja funcionando corretamente antes de avançar para a seguinte**.

---

## 📖 Referências principais

* PI-CAI — Prostate Imaging: Cancer AI.
* Grand Challenge.
* Ronneberger, O.; Fischer, P.; Brox, T. — *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
* Chen, T.; Guestrin, C. — *XGBoost: A Scalable Tree Boosting System*.
* van Griethuysen, J. J. M. et al. — *Computational Radiomics System to Decode the Phenotype of Tumors*.

---

## ⚠️ Observação

Este projeto possui finalidade **acadêmica e experimental**. Os modelos desenvolvidos não devem ser considerados ferramentas clínicas ou utilizados para diagnóstico médico sem validação apropriada, aprovação ética e avaliação clínica independente.
