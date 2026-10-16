# Projeto 3 — Variational Autoencoder (VAE)

Projeto desenvolvido para a Unidade Curricular de **Aprendizado de Máquina**, sob orientação do professor **Marcos G. Quiles**, na Universidade Federal de São Paulo (UNIFESP).

## Objetivo

Implementar e avaliar modelos **Variational Autoencoder (VAE)** para explorar a representação de dados em espaços latentes de diferentes dimensões, investigando a formação de clusters, a separação de classes e a preservação das informações originais.

## Desenvolvimento

O projeto contempla as seguintes etapas:

1. **Seleção dos datasets**
   - Selecionar dois datasets rotulados para treinamento e avaliação dos modelos.

2. **Treinamento e otimização dos VAEs**
   - Implementar e treinar diferentes topologias de VAE.
   - Selecionar a melhor arquitetura com base na função de custo obtida no conjunto de validação.
   - Considerar espaços latentes bidimensionais (2D) e de dimensões superiores.

3. **Exploração do espaço latente**
   - Visualizar diretamente os espaços latentes 2D.
   - Aplicar técnicas de redução de dimensionalidade, como PCA, UMAP ou t-SNE, para projetar espaços latentes de maior dimensão em 2D.
   - Utilizar os rótulos originais para identificar e comparar as classes nas visualizações.

## Análise dos resultados

A análise busca responder às seguintes questões:

- **Formação de clusters:** os dados apresentam agrupamentos bem definidos no espaço latente?
- **Separação das classes:** os rótulos originais apresentam uma separação clara nas representações aprendidas?
- **Variância explicada:** quanto da variância dos dados latentes é preservado pelas projeções 2D, especialmente utilizando PCA?

## Etapa adicional (opcional)

Investigar estratégias para orientar a formação do espaço latente utilizando os rótulos disponíveis, buscando melhorar a separação entre classes e a organização dos agrupamentos.
