# ROADMAP — PI-CAI: Segmentação, Radiômica e Classificação

## 1. Visão geral

Este projeto tem como objetivo desenvolver e avaliar uma pipeline híbrida para detecção de câncer de próstata clinicamente significativo (csPCa) utilizando exames de ressonância magnética do dataset PI-CAI.

A pipeline será composta pelas seguintes etapas principais:

```text
PI-CAI
  │
  ▼
Organização dos dados
  │
  ▼
Pré-processamento
  │
  ▼
Segmentação com U-Net
  │
  ▼
Máscaras das regiões de interesse
  │
  ▼
Extração de características radiômicas
  │
  ▼
Seleção de features
  │
  ▼
Classificação com XGBoost
  │
  ▼
Avaliação
  │
  ▼
Análise dos resultados
```

---

## 2. Etapa 01 — Organização dos dados

### Objetivo

Estruturar o dataset PI-CAI e os arquivos de labels de forma que possam ser acessados programaticamente.

### Atividades

* Organizar os cinco folds do PI-CAI.
* Organizar o repositório de labels.
* Identificar a relação entre `patient_id`, `study_id` e os identificadores dos exames.
* Mapear imagens e máscaras.
* Conferir a existência dos arquivos necessários.
* Criar um carregador inicial dos dados.

### Entregável

Um módulo capaz de localizar e carregar automaticamente as imagens, máscaras e informações clínicas correspondentes a cada caso.

---

## 3. Etapa 02 — Exploração e validação dos dados

### Objetivo

Confirmar que os arquivos estão corretamente relacionados antes de iniciar o treinamento.

### Atividades

* Inspecionar os arquivos `.mha` e `.nii.gz`.
* Verificar dimensões dos volumes.
* Verificar espaçamento e orientação.
* Conferir os sistemas de coordenadas.
* Visualizar imagens individualmente.
* Sobrepor imagens e máscaras.
* Identificar possíveis casos problemáticos.

### Entregável

Visualização validada de casos contendo imagem + máscara correspondente.

> Esta etapa deve ser concluída antes do treinamento da U-Net.

---

## 4. Etapa 03 — Pré-processamento

### Objetivo

Preparar as imagens para entrada na rede neural.

### Atividades

* Padronização das imagens.
* Normalização das intensidades.
* Alinhamento, quando necessário.
* Definição da região de interesse.
* Conversão dos volumes 3D para cortes 2D.
* Associação entre cada slice e sua respectiva máscara.
* Divisão dos dados em treinamento, validação e teste.

### Entregável

Dataset pronto para treinamento da U-Net.

---

## 5. Etapa 04 — Segmentação com U-Net

### Objetivo

Desenvolver um modelo capaz de realizar a segmentação automática das regiões de interesse.

### Atividades

* Implementar a U-Net 2D.
* Definir função de perda.
* Configurar treinamento.
* Utilizar GPU RTX 3070.
* Monitorar treinamento e validação.
* Avaliar diferentes hiperparâmetros.
* Salvar o melhor modelo.

### Métricas

* Dice Score
* IoU

### Entregável

Modelo U-Net treinado e capaz de gerar máscaras automaticamente.

---

## 6. Etapa 05 — Geração e validação das máscaras

### Objetivo

Utilizar a U-Net treinada para produzir as máscaras necessárias à etapa radiômica.

### Atividades

* Gerar máscaras para os casos selecionados.
* Reconstruir os volumes, quando necessário.
* Verificar visualmente as predições.
* Comparar máscaras previstas e referências.
* Identificar casos de baixa qualidade.

### Entregável

Conjunto de imagens e máscaras segmentadas pronto para a extração radiômica.

---

## 7. Etapa 06 — Extração Radiômica

### Objetivo

Transformar as informações presentes nas imagens em características quantitativas.

### Categorias inicialmente consideradas

* Características de primeira ordem.
* Características de forma.
* Características de textura.

### Atividades

* Configurar o PyRadiomics.
* Definir as imagens utilizadas.
* Aplicar as máscaras.
* Extrair as features.
* Organizar os resultados em uma tabela.
* Verificar valores ausentes e inconsistentes.

### Entregável

Dataset tabular contendo as características radiômicas e os respectivos labels clínicos.

---

## 8. Etapa 07 — Seleção de Features

### Objetivo

Reduzir dimensionalidade e remover características redundantes.

### Atividades

* Analisar correlação entre features.
* Remover atributos altamente correlacionados.
* Avaliar métodos de seleção.
* Considerar RFE e/ou importância das features do XGBoost.
* Garantir que a seleção seja realizada sem data leakage.

### Entregável

Conjunto reduzido de features para treinamento do classificador.

---

## 9. Etapa 08 — Classificação com XGBoost

### Objetivo

Classificar os casos de acordo com a presença de câncer de próstata clinicamente significativo.

### Atividades

* Preparar os dados de entrada.
* Treinar o XGBoost.
* Ajustar hiperparâmetros.
* Obter probabilidades de classificação.
* Analisar a importância das features.
* Definir o threshold de decisão.

### Entregável

Modelo de classificação treinado e avaliado.

---

## 10. Etapa 09 — Avaliação

### Segmentação

Avaliar:

* Dice Score
* IoU

### Classificação

Avaliar:

* ROC-AUC
* Sensibilidade
* Especificidade
* F1-Score
* Matriz de confusão

### Estratégia

Utilizar validação cruzada e manter separação adequada entre treinamento e validação para evitar data leakage.

### Entregável

Tabela consolidada de resultados e métricas.

---

## 11. Etapa 10 — Baselines

### Objetivo

Comparar o XGBoost com modelos tradicionais.

### Modelos candidatos

* Random Forest
* SVM

Os modelos deverão utilizar, sempre que possível, o mesmo conjunto de features e a mesma estratégia de validação.

### Entregável

Comparação entre os modelos propostos.

---

## 12. Etapa 11 — Análise dos resultados

### Objetivo

Interpretar os resultados obtidos e identificar limitações da abordagem.

### Análises

* Desempenho da segmentação.
* Desempenho da classificação.
* Comparação com baselines.
* Features mais relevantes.
* Casos de erro.
* Falsos positivos.
* Falsos negativos.
* Impacto da seleção de features.
* Custo computacional.

### Entregável

Discussão dos resultados e das limitações da pipeline.

---

## 13. Etapa 12 — Documentação e TCC II

### Objetivo

Consolidar o desenvolvimento e os resultados no trabalho final.

### Atividades

* Documentar experimentos.
* Organizar gráficos e tabelas.
* Registrar configurações dos modelos.
* Descrever resultados.
* Elaborar discussão.
* Redigir conclusões.
* Organizar código e documentação do projeto.

### Entregável

Versão final do TCC e repositório organizado.

---

# Visão resumida do projeto

```text
[01] ORGANIZAÇÃO
       ↓
[02] EXPLORAÇÃO DOS DADOS
       ↓
[03] PRÉ-PROCESSAMENTO
       ↓
[04] U-NET
       ↓
[05] MÁSCARAS
       ↓
[06] RADIÔMICA
       ↓
[07] SELEÇÃO DE FEATURES
       ↓
[08] XGBOOST
       ↓
[09] AVALIAÇÃO
       ↓
[10] BASELINES
       ↓
[11] ANÁLISE
       ↓
[12] TCC II
```

# Prioridade de desenvolvimento

### 🔴 Alta prioridade

1. Organização dos dados
2. Validação das imagens e máscaras
3. Pré-processamento
4. U-Net

### 🟡 Média prioridade

5. Geração das máscaras
6. Radiômica
7. Seleção de features
8. XGBoost

### 🟢 Etapa final

9. Avaliação
10. Baselines
11. Análise
12. Documentação

---

## Regra principal do projeto

**Não avançar para a próxima etapa sem validar a anterior.**

Especialmente:

```text
Imagens corretas
      ↓
Máscaras corretas
      ↓
U-Net funcionando
      ↓
Features radiômicas válidas
      ↓
XGBoost
      ↓
Avaliação
```

Isso reduz bastante o risco de chegar ao final do projeto com um resultado ruim sem saber em qual etapa o problema surgiu.
