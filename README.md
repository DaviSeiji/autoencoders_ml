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
internet: Dry Bean é obtido da UCI via `ucimlrepo` e MNIST via `torchvision`.
Os dados brutos ficam em `data/` (ignorado pelo Git) para reutilização.

Etapas iniciais implementadas no notebook:

- [x] Importar as bibliotecas e configurar seeds para reprodutibilidade.
- [x] Carregar Dry Bean e MNIST diretamente pelo Python.
- [x] Separar treino, validação e teste com estratificação e seed fixa.
- [x] Padronizar Dry Bean usando apenas o treino e normalizar MNIST para `[0, 1]`.
- [x] Definir um VAE denso reutilizável, com topologia e dimensão latente configuráveis.
- [x] Implementar busca reutilizável com Optuna/TPE, persistência SQLite e retomada.
- [x] Preparar painéis 2×2 dos latentes 2D/8D/16D/32D e MSE por classe.

O notebook oferece `criar_vae` e `treinar_vae` para os dois datasets, com
custo de reconstrução + peso KL × KL, em PyTorch.
Cada execução registra as perdas total, de reconstrução e KL de treino e
validação em `results/historicos/`. A parada antecipada e a escolha dos pesos
usam `val_score = reconstrução + KL`, comum a todos os pesos KL testados.
Os pesos são salvos em `results/modelos/`, em arquivos `.pt`.

O notebook contém as seções **Otimização de Hiperparâmetros — Optuna** e
**Experimento 1 — Dry Bean (Latente 2D)**. A busca usa somente treino e
validação, mantém a dimensão latente fixa e permite configurar o espaço de
busca. Cada estudo usa 30 trials. Dry Bean mantém até 100 épocas e patience 10;
MNIST usa até 50 épocas e patience 7. A busca testa learning rate de `1e-5`
a `1e-2`, ReLU/Tanh/LeakyReLU, Adam/RMSprop/SGD e peso KL 0,1/0,5/1/2,
além das camadas, neurônios e batch sizes já utilizados.

Cada estudo fica em `results/optuna/<nome_estudo>/`, com SQLite, tabela dos
trials, melhor configuração em JSON, histórico com as seis perdas e pesos
do melhor modelo. O estado do sampler também é salvo para retomada.
As células de busca exibem o top 5 de cada estudo, com as topologias do encoder
e decoder. Ao final de cada dataset, uma tabela mostra o melhor trial de
2D, 8D, 16D e 32D. Logs informativos ficam ocultos; avisos e erros permanecem visíveis.
`n_trials` indica o total desejado, sem repetir trials finalizados. Reutilize
o mesmo nome apenas com os mesmos dados e configuração. A função
`carregar_melhor_vae` recria a rede selecionada e carrega os pesos salvos.
O salvamento por trial fica desativado durante a busca; somente os artefatos
da melhor configuração são mantidos. As chamadas para Dry Bean 8D, 16D e
32D estão prontas em células separadas, com 30 trials cada, seguidas de uma
tabela comparativa. Também estão prontas as chamadas do MNIST em 2D, 8D,
16D e 32D, com a mesma meta de 30 trials e uma tabela comparativa própria.
Essas novas buscas ainda não foram executadas. As células do MNIST usam
as imagens normalizadas 28 × 28, sem usar o conjunto de teste, e salvam
os resultados em `results/optuna/mnist_<dimensão>d/`.

Os estudos PyTorch devem ser executados do início. Resultados TensorFlow
anteriores ficam arquivados em `results/anteriores_tensorflow/` e as saídas
antigas do notebook foram limpas. Reinicie o kernel e execute em ordem.

A seed utilizada é `42`. A reprodução dos experimentos também depende das
mesmas versões das bibliotecas e do mesmo ambiente de execução; o notebook
configura operações determinísticas. Após verificar ausentes e remover duplicatas,
Dry Bean usa a proporção 70%/15%/15% para
treino/validação/teste, com `StandardScaler` ajustado somente no treino.
MNIST preserva o teste oficial e separa 10.000 das 60.000 imagens de treino
para validação. As imagens normalizadas mantêm o formato 28 × 28.
Os rótulos são preservados para análise, e tabelas comparam as porcentagens
das classes entre os conjuntos.

As funções são verificadas com pequenos dados sintéticos, incluindo retomada
do SQLite e recarga dos pesos; os estudos finais ficam para execução pelo usuário.

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

A seção **Visualização dos espaços latentes — PCA e MSE** carrega os melhores
pesos de cada estudo, sem repetir a busca. Para cada dataset, mostra painéis
2×2 e barras de MSE por classe, primeiro para validação e depois para teste. O latente
2D é mostrado diretamente; 8D, 16D e 32D são projetados em 2D. O PCA é
ajustado somente nos latentes de treino. Cada figura informa o MSE de
reconstrução de todo o conjunto indicado, usando `decoder(z_mean)`, sem KL
e sem multiplicar pelo número de atributos/pixels. As cores das classes são
consistentes. São desenhados até 5.000 exemplos por conjunto com amostragem
estratificada. As figuras PNG e os CSVs com MSE e variância explicada ficam
em `results/figuras/`.

Ao final do notebook, a seção **Exemplo de reconstrução** compara a primeira
instância de teste com as reconstruções dos quatro modelos. Para MNIST, exibe
a imagem original e as quatro reconstruções lado a lado. Para Dry Bean, mostra
um gráfico dos atributos padronizados e uma tabela nas unidades originais.
O índice do exemplo pode ser alterado nas células. Os gráficos usam os pesos
salvos e a média do encoder, sem novo treinamento.

A análise busca responder às seguintes questões:

- **Formação de clusters:** os dados apresentam agrupamentos bem definidos no espaço latente?
- **Separação das classes:** os rótulos originais apresentam uma separação clara nas representações aprendidas?
- **Variância explicada:** quanto da variância dos dados latentes é preservado pelas projeções 2D, especialmente utilizando PCA?

## Etapa adicional (opcional)

Investigar estratégias para orientar a formação do espaço latente utilizando os rótulos disponíveis, buscando melhorar a separação entre classes e a organização dos agrupamentos.
