# Projeto 3 — Variational Autoencoder (VAE)

Projeto desenvolvido para a Unidade Curricular de **Aprendizado de Máquina**, sob orientação do professor **Marcos G. Quiles**, na Universidade Federal de São Paulo (UNIFESP).

## Objetivo

Implementar e avaliar modelos **Variational Autoencoder (VAE)** para explorar a representação de dados em espaços latentes de diferentes dimensões, investigando a formação de clusters, a separação de classes e a preservação das informações originais.

## Desenvolvimento

### Execução do notebook

Instale as dependências no Python que você usa (PowerShell):

```powershell
python -m pip install -r requirements.txt
```

Abra `notebooks/projeto_vae.ipynb` no VS Code, selecione esse mesmo Python
como kernel e execute as células em ordem. O primeiro carregamento requer
internet: Dry Bean é obtido da UCI via `ucimlrepo` e MNIST via TensorFlow/Keras.
Os dados brutos ficam em `data/` (ignorado pelo Git) para reutilização.

Etapas iniciais implementadas no notebook:

- [x] Importar as bibliotecas e configurar seeds para reprodutibilidade.
- [x] Carregar Dry Bean e MNIST diretamente pelo Python.
- [x] Separar treino, validação e teste com estratificação e seed fixa.
- [x] Padronizar Dry Bean usando apenas o treino e normalizar MNIST para `[0, 1]`.
- [x] Definir um VAE denso reutilizável, com topologia e dimensão latente configuráveis.

O notebook oferece `criar_vae` e `treinar_vae` para os dois datasets, com
custo de reconstrução + KL e parada antecipada baseada em `val_loss`.
Os exemplos de uso estão em Markdown; os treinamentos e a comparação
de arquiteturas ainda devem ser executados.

A seed utilizada é `42`. A reprodução dos experimentos também depende das
mesmas versões das bibliotecas e do mesmo ambiente de execução; o notebook
configura operações determinísticas. Após verificar ausentes e remover duplicatas,
Dry Bean usa a proporção 70%/15%/15% para
treino/validação/teste, com `StandardScaler` ajustado somente no treino.
MNIST preserva o teste oficial e separa 10.000 das 60.000 imagens de treino
para validação. As imagens normalizadas mantêm o formato 28 × 28.
Os rótulos são preservados para análise, e tabelas comparam as porcentagens
das classes entre os conjuntos.

As células de pré-processamento estão preparadas para execução pelo usuário;
a validação durante o desenvolvimento usa apenas pequenos dados sintéticos.

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
