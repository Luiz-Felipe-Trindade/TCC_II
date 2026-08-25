# TODO — PI-CAI: Segmentação, Radiômica e Classificação

## Status

* [ ] Projeto configurado
* [ ] Dataset organizado
* [ ] Pipeline de dados validada
* [ ] U-Net implementada
* [ ] Radiômica implementada
* [ ] XGBoost implementado
* [ ] Avaliação concluída
* [ ] Resultados documentados

---

# 1. Configuração do ambiente

* [ ] Criar repositório do projeto
* [ ] Criar ambiente virtual Python
* [ ] Configurar VS Code
* [ ] Configurar Git
* [ ] Criar `.gitignore`
* [ ] Criar `README.md`
* [ ] Criar `ROADMAP.md`
* [ ] Criar `TODO.md`
* [ ] Registrar versões das principais bibliotecas
* [ ] Verificar funcionamento da RTX 3070 com CUDA
* [ ] Testar PyTorch utilizando GPU

---

# 2. Organização do dataset

* [ ] Confirmar download dos 5 folds do PI-CAI
* [ ] Organizar os folds na pasta `data/raw`
* [ ] Organizar o repositório de labels
* [ ] Conferir `additional_resources`
* [ ] Conferir `anatomical_delineations`
* [ ] Conferir `clinical_information`
* [ ] Conferir `csPCa_lesion_delineations`
* [ ] Verificar o arquivo `ProstateX-mapping.json`
* [ ] Verificar o arquivo de informações clínicas
* [ ] Identificar a relação entre `patient_id` e `study_id`
* [ ] Identificar a relação entre imagens e máscaras
* [ ] Documentar o padrão de nomenclatura dos arquivos

---

# 3. Exploração dos dados

* [ ] Criar script para listar os casos
* [ ] Criar função para localizar um exame
* [ ] Carregar um arquivo `.mha`
* [ ] Carregar um arquivo `.nii.gz`
* [ ] Verificar dimensões dos volumes
* [ ] Verificar espaçamento dos voxels
* [ ] Verificar orientação dos volumes
* [ ] Verificar quantidade de slices
* [ ] Verificar existência de valores inválidos
* [ ] Visualizar imagens T2W
* [ ] Visualizar imagens ADC
* [ ] Visualizar imagens DWI/HBV
* [ ] Visualizar máscaras anatômicas
* [ ] Visualizar máscaras de lesões

---

# 4. Validação imagem × máscara

> **Importante:** esta etapa deve ser concluída antes do treinamento da U-Net.

* [ ] Selecionar casos de teste
* [ ] Sobrepor máscara e imagem
* [ ] Verificar se imagem e máscara estão espacialmente alinhadas
* [ ] Verificar orientação
* [ ] Verificar resolução
* [ ] Verificar correspondência entre slices
* [ ] Identificar casos desalinhados
* [ ] Identificar casos sem máscara
* [ ] Registrar problemas encontrados
* [ ] Definir procedimento para casos inválidos

---

# 5. Pré-processamento

* [ ] Definir quais sequências serão utilizadas
* [ ] Definir estratégia de normalização
* [ ] Definir estratégia de redimensionamento
* [ ] Definir tamanho dos inputs da U-Net
* [ ] Definir estratégia para conversão 3D → 2D
* [ ] Criar rotina para geração dos slices
* [ ] Associar cada slice à sua máscara
* [ ] Verificar slices sem região de interesse
* [ ] Definir estratégia para esses slices
* [ ] Criar dataset para treinamento
* [ ] Criar dataset para validação
* [ ] Criar dataset para teste
* [ ] Garantir separação por paciente
* [ ] Verificar possibilidade de data leakage

---

# 6. U-Net

## Implementação

* [ ] Criar arquitetura U-Net 2D
* [ ] Definir número de canais de entrada
* [ ] Definir número de classes de saída
* [ ] Definir função de perda
* [ ] Definir otimizador
* [ ] Definir learning rate
* [ ] Definir batch size
* [ ] Definir número inicial de épocas
* [ ] Implementar treinamento
* [ ] Implementar validação
* [ ] Implementar checkpoint
* [ ] Implementar early stopping, se necessário

## Primeiro experimento

* [ ] Treinar modelo inicial
* [ ] Monitorar loss
* [ ] Monitorar Dice
* [ ] Monitorar IoU
* [ ] Salvar melhor modelo
* [ ] Avaliar no conjunto de teste
* [ ] Visualizar predições

## Ajustes

* [ ] Analisar underfitting
* [ ] Analisar overfitting
* [ ] Ajustar hiperparâmetros
* [ ] Testar função de perda alternativa, se necessário
* [ ] Comparar experimentos
* [ ] Escolher modelo final

---

# 7. Geração das máscaras

* [ ] Carregar U-Net treinada
* [ ] Gerar máscaras para os casos selecionados
* [ ] Salvar máscaras previstas
* [ ] Manter correspondência com `study_id`
* [ ] Verificar visualmente as máscaras
* [ ] Comparar máscaras previstas e referências
* [ ] Registrar casos problemáticos
* [ ] Definir quais máscaras serão utilizadas na radiômica

---

# 8. Radiômica

* [ ] Configurar PyRadiomics
* [ ] Definir imagens de entrada
* [ ] Definir ROI
* [ ] Definir configuração de extração
* [ ] Testar extração em um caso
* [ ] Verificar features geradas
* [ ] Identificar features de primeira ordem
* [ ] Identificar features de forma
* [ ] Identificar features de textura
* [ ] Executar extração em lote
* [ ] Salvar resultados em formato tabular
* [ ] Verificar valores ausentes
* [ ] Verificar valores infinitos
* [ ] Verificar features constantes
* [ ] Documentar configuração utilizada

---

# 9. Seleção de features

* [ ] Analisar dimensionalidade inicial
* [ ] Calcular correlação entre features
* [ ] Remover atributos altamente correlacionados
* [ ] Avaliar features redundantes
* [ ] Definir método final de seleção
* [ ] Testar RFE, se aplicável
* [ ] Avaliar importância das features pelo XGBoost
* [ ] Registrar quantidade de features antes e depois
* [ ] Garantir seleção sem data leakage

---

# 10. Dataset para classificação

* [ ] Associar features radiômicas aos labels clínicos
* [ ] Associar cada caso ao `case_csPCa`
* [ ] Verificar classes positivas e negativas
* [ ] Avaliar desbalanceamento
* [ ] Definir estratégia para desbalanceamento, se necessária
* [ ] Separar treino e teste
* [ ] Garantir separação por paciente
* [ ] Garantir ausência de data leakage

---

# 11. XGBoost

## Modelo inicial

* [ ] Implementar XGBoost
* [ ] Treinar modelo inicial
* [ ] Gerar probabilidades
* [ ] Gerar classificações
* [ ] Calcular métricas
* [ ] Registrar configuração utilizada

## Ajuste

* [ ] Avaliar hiperparâmetros
* [ ] Avaliar regularização
* [ ] Avaliar overfitting
* [ ] Analisar importância das features
* [ ] Definir modelo final

## Threshold

* [ ] Gerar curva ROC
* [ ] Gerar curva Precision-Recall
* [ ] Avaliar diferentes thresholds
* [ ] Analisar sensibilidade
* [ ] Analisar especificidade
* [ ] Definir threshold final
* [ ] Justificar escolha

---

# 12. Baselines

* [ ] Implementar Random Forest
* [ ] Implementar SVM
* [ ] Utilizar o mesmo conjunto de features
* [ ] Utilizar a mesma estratégia de validação
* [ ] Calcular as mesmas métricas
* [ ] Comparar resultados com XGBoost

---

# 13. Validação

* [ ] Definir estratégia final de validação
* [ ] Implementar 5-Fold Cross Validation, se confirmado
* [ ] Utilizar divisão estratificada
* [ ] Garantir separação por paciente
* [ ] Verificar centros de aquisição
* [ ] Evitar vazamento entre folds
* [ ] Realizar seleção de features dentro do treinamento
* [ ] Registrar resultados de cada fold
* [ ] Calcular média
* [ ] Calcular desvio padrão

---

# 14. Avaliação da segmentação

* [ ] Calcular Dice Score
* [ ] Calcular IoU
* [ ] Avaliar distribuição dos resultados
* [ ] Identificar casos com pior desempenho
* [ ] Visualizar exemplos de erro
* [ ] Registrar resultados finais

---

# 15. Avaliação da classificação

* [ ] Calcular ROC-AUC
* [ ] Calcular sensibilidade
* [ ] Calcular especificidade
* [ ] Calcular F1-Score
* [ ] Gerar matriz de confusão
* [ ] Gerar curva ROC
* [ ] Gerar curva Precision-Recall
* [ ] Avaliar falsos positivos
* [ ] Avaliar falsos negativos
* [ ] Comparar XGBoost e baselines

---

# 16. Análise dos resultados

* [ ] Identificar melhores configurações
* [ ] Identificar principais features
* [ ] Avaliar impacto da seleção de features
* [ ] Avaliar impacto da segmentação
* [ ] Analisar erros de classificação
* [ ] Analisar limitações
* [ ] Avaliar custo computacional
* [ ] Registrar tempo de treinamento
* [ ] Registrar uso de GPU
* [ ] Comparar resultados com literatura/PI-CAI

---

# 17. Reprodutibilidade

* [ ] Salvar configurações dos experimentos
* [ ] Registrar seeds
* [ ] Registrar versões das bibliotecas
* [ ] Registrar hardware utilizado
* [ ] Registrar parâmetros dos modelos
* [ ] Organizar checkpoints
* [ ] Organizar resultados
* [ ] Criar scripts reproduzíveis
* [ ] Atualizar README

---

# 18. Documentação do TCC II

* [ ] Documentar metodologia implementada
* [ ] Documentar pré-processamento
* [ ] Documentar U-Net
* [ ] Documentar radiômica
* [ ] Documentar XGBoost
* [ ] Documentar validação
* [ ] Criar tabelas de resultados
* [ ] Criar gráficos
* [ ] Criar discussão
* [ ] Documentar limitações
* [ ] Elaborar conclusões
* [ ] Revisar referências
* [ ] Revisar código
* [ ] Finalizar TCC

---

# Marco principal do projeto

O projeto será considerado funcional quando for possível executar:

```text
Imagem PI-CAI
      ↓
Pré-processamento
      ↓
U-Net
      ↓
Máscara
      ↓
Radiômica
      ↓
Features selecionadas
      ↓
XGBoost
      ↓
Probabilidade de csPCa
      ↓
Métricas de avaliação
```

## Regra de desenvolvimento

**Não otimizar uma etapa antes de confirmar que a etapa anterior está funcionando corretamente.**

Primeiro:

```text
"Funciona?"
```

Depois:

```text
"Está correto?"
```

E somente então:

```text
"Como posso melhorar?"
```
