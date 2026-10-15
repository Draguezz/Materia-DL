# Capítulo 3

Conteúdo do capítulo 3.

## Fórmula Geral da Regressão Linear Simples
Na regressão linear simples, temos a seguinte fórmula:
` y = a . x + b`

onde:

- `y` é a variável dependente ou alvo (o valor que queremos prever).

- `x` é a variável independente ou preditor (o valor de entrada).

- `a` é o peso, também conhecido como coeficiente angular, que define a inclinação da reta. O coeficiente angular 
 pode ser calculado pela razão entre a variação de 
 e a variação de 
, ou seja:
(adicionar img)
 
Este valor indica o quanto a variável dependente `y` varia para cada mudança de uma unidade em `x`
. Se `a`
 for positivo, a reta sobe conforme `x`
 aumenta. Se for negativo, a reta desce.

 - `b` é o intercepto (ou bias), que determina onde a reta cruza o eixo Y (o valor de `y`
 quando `x` = 0
).

O objetivo da regressão linear é ajustar os parâmetros 
 `a` e `b` 
 de modo que a reta resultante minimize a diferença entre as previsões 
 `y`previsto e os valores `y` reais 
. Essa diferença é quantificada por uma função de erro.


## A Relação entre a Regressão Linear e um Neurônio
A equação da regressão linear se assemelha muito à equação que define o funcionamento de um neurônio em uma rede neural. No caso de um neurônio, temos:

ô
onde:

ô
 é a saída do neurônio.

 é o valor de entrada.

 é o peso, que funciona de forma semelhante à inclinação 
 na regressão linear, indicando a importância de 
 na saída 
ô
.

 é o bias, que ajusta a saída do neurônio, assim como o intercepto na regressão linear.

Essa equação é idêntica à da regressão linear. A diferença no neurônio é que, após calcular o valor 

## Rede Neural como Aproximadora de Funções Não Lineares

import numpy as np  

import matplotlib.pyplot as plt 

import tensorflow as tf 

from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense 

