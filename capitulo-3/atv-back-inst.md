Implemente o gradiente descendente à mão, sem biblioteca de rede neural, com as fórmulas da seção anterior. Escreva o treino como uma função que recebe a taxa e devolve o número de iterações até o erro ficar abaixo de ERRO_ALVO e a lista de erros do caminho. Sorteie w e b uma vez, como na seção anterior, logo depois da nuvem, e comece todas as taxas desse mesmo ponto: sem isso, a comparação mede também a sorte do sorteio. O erro da primeira iteração deve ser 
. Se o laço chegar a LIMITE, pare e avise. Entregue:

a tabela taxa → iterações, com as três linhas, e as três curvas de erro num mesmo par de eixos, com escala logarítmica no eixo das iterações, porque a curva mais lenta é cerca de cem vezes mais longa que a mais rápida;

cada contagem da tabela dividida pela seguinte, que dá aproximadamente o mesmo número nas duas divisões, e uma frase dizendo por que esse número aparece;

com ERRO_ALVO = 0.001, que faz as três taxas pararem pelo LIMITE, o menor erro que as curvas alcançam, com quatro casas e marcado no gráfico por uma linha horizontal tracejada;

a comparação desse piso com np.var(y_real - (4 + 3 * x)), a variância do ruído que gerou os dados, e o motivo de os dois números coincidirem;

com ERRO_ALVO de volta em 1.1 e as taxas 0.3 e 0.5 acrescentadas à lista, as cinco curvas, agora com escala logarítmica também no eixo do erro e o eixo das iterações limitado às 100 primeiras, e qual delas sobe e o que acontece com ela se o laço continuar.

Resultados Esperados: As duas divisões dão cerca de dez: dividir a taxa por dez multiplica as iterações por aproximadamente dez. O piso e a variância do ruído diferem em menos de um milésimo, e é essa coincidência que responde a pergunta, porque o erro que sobra não é da reta, é do ruído dos dados: nenhuma reta consegue furá-lo. Das cinco curvas, a de 0.5 cresce desde a primeira iteração e passa de 
 antes da centésima. Se o laço continuar, o erro estoura para inf e depois vira nan, porque as contas passam a subtrair infinito de infinito, e o laço só para porque existe o LIMITE. As duas pontas juntas, o piso que não se fura e o passo grande demais, são o motivo de a taxa de aprendizado ser um parâmetro que se escolhe com cuidado.
