# Ativando ambiente
import Pkg
Pkg.activate("C:/Users/93dav/programacao_nao_linear")

# Importando libs que serão utilizadas
import Pkg; Pkg.add("Optim")
import Pkg; Pkg.add("CUTEst")
import Pkg; Pkg.add("NLPModels")
import Pkg; Pkg.add("LinearAlgebra")
import Pkg; Pkg.add("DataFrames")

using Optim, CUTEst, NLPModels, LinearAlgebra, DataFrames

# Métricas de desempenho
mutable struct metrics
    function_value::Float64
    gradient_norm_value::Float64
    function_count::Int
    gradient_count::Int
    iteracoes::Int
    tempo::Float64
end

# Criando DataFrame para armazenar os resultados
results_df = DataFrame(
    Problem = String[],
    N_var = Union{Float64, Missing}[],
    Densidade = Union{Float64, Missing}[],
    Method = String[],
    Gradiente = Union{Float64, Missing}[],
    Funcao = Union{Float64, Missing}[],
    Iteracoes = Int[],
    Avaliacoes_grad = Union{Float64, Missing}[],
    Avaliacoes_f = Union{Float64, Missing}[],
    Max_iteracoes = Int[],
    Fator_reducao = Float64[],  # Alterado para Float64
    Passo_inicial_armijo = Float64[],  # Alterado para Float64
    Parada_grad = Float64[],
    Parada_armijo = Float64[],
    Tempo = Union{Float64, Missing}[],
    Memoria = Union{Float64, Missing}[]
)

#================================================================================================================
`registro!(results_df, nlp, count::metrics, algoritmo::String, iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, tempo, memoria)`

# Objetivo
    Registra os resultados de cada execução no DataFrame `results_df`.

# Saída
    - Armazena os dados da execução (função objetivo, gradiente, iterações, etc.) no DataFrame para análise posterior.

# Entradas
    - `results_df`       : DataFrame onde serão registrados os dados.
    - `nlp`              : Objeto do problema de teste.
    - `count`            : Estrutura com métricas do problema.
    - `algoritmo`        : Nome do algoritmo utilizado.
    - `iteracoes`        : Número de iterações realizadas.
    - `fator_reducao`    : Fator de redução utilizado na busca de Armijo.
    - `passo_inicial`    : Passo inicial para a busca de Armijo.
    - `parada_grad`      : Critério de parada para a norma do gradiente.
    - `parada_armijo`    : Critério de parada para a busca de Armijo.
    - `tempo`            : Tempo de execução da função.
    - `memoria`          : Memória alocada durante a execução.
=================================================================================================================#
function registro!(results_df, nlp, count::metrics, algoritmo::String, iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, tempo, memoria)
    # Obter o nome do problema atual
    problem_name = nlp.meta.name

    # Adiciona uma nova linha com todas as informações
    push!(results_df, (
        Problem = problem_name,
        N_var = nlp.meta.nvar,
        Densidade = nlp.meta.nnzh/(nlp.meta.nvar^2),
        Method = algoritmo,
        Gradiente = count.gradient_norm_value,
        Funcao = count.function_value,
        Iteracoes = iteracoes,
        Avaliacoes_grad = count.gradient_count,
        Avaliacoes_f = count.function_count,
        Max_iteracoes = iteracoes,
        Fator_reducao = fator_reducao,
        Passo_inicial_armijo = passo_inicial,
        Parada_grad = parada_grad,
        Parada_armijo = parada_armijo,
        Tempo = tempo,
        Memoria = memoria
    ))
end



#================================================================================================================
`armijo_search(nlp, x, d, passo_inicial, fator_reducao, η, count)`

# Objetivo
    Realizar a busca de Armijo para definir o tamanho do passo `α`.

# Saída
    - Retorna o valor de `α`, tamanho do passo ajustado pela busca de Armijo.

# Entradas
    - `nlp`             : Problema de teste importado pela função `CUTEstModel()`.
    - `x`               : Ponto atual da busca.
    - `d`               : Direção de descida.
    - `passo_inicial`   : Passo inicial da busca de Armijo.
    - `fator_reducao`   : Fator de redução aplicado na busca de Armijo.
    - `η`               : Parâmetro da condição de Armijo.
    - `count`           : Estrutura para acumular métricas de desempenho.

=================================================================================================================#
function armijo_search(nlp, x, d, passo_inicial, fator_reducao, η, count)
    α = passo_inicial
    k = 0
    count.function_count += 2
    while (obj(nlp, x + α * d) > obj(nlp, x) - η * α * d' * d) && k < 50
        count.function_count += 2
        α *= fator_reducao
        k += 1
    end
    return α
end



#================================================================================================================
`Cauchy(nlp, x, d, passo_inicial, fator_reducao, cont, criterio_parada)`

# Objetivo
    Aplicar o algoritmo do método do gradiente com busca clássica de Armijo

# Saída
    - Print das iterações com os valores da função e do gradiente
    - Ajusta o valor da estrutura count para anotar:
        1. Valor da função no ponto final
        2. Norma do gradiente no ponto final
        3. Quantidade de vezes que a função foi avaliada
        4. Quantidade de vezes que o gradiente foi avaliado

# Entradas
    - `nlp`             : problema teste importado pela função CUTEstModel() 
    - `x`               : x inicial da busca. Sugestão: puxar o que está na variável nlp.meta.x0
    - `d`               : gradiente da função no ponto inicial. Sugestão: -grad(nlp,x). A variável d foi usada para economizar 1 cálulo do gradiente por iteração.
    - `passo_inicial`   : passo inicial da busca de Armijo. Sugesão: 1, 0.5 ou 0.1
    - `fator_reducao`   : fator de redução do passo a cada teste da Busca de Armijo. Número entre 0 e 1. Sugestão: 0.5
    - `cont`            : estrutura que acumula as métricas do problema
    - `criterio_parada` : a partir de qual valor da norma do gradiente o algoritmo pode parar
=================================================================================================================#
function Cauchy(nlp, x, d, passo_inicial, fator_reducao, η, count, criterio_parada_grad, criterio_parada_i)
    i = 0
    while (norm(d) > criterio_parada_grad) && (i < criterio_parada_i)
        # Definindo o gradiente
        d = -grad(nlp, x)
        count.gradient_count += 1

        # Busca de Armijo
        α = armijo_search(nlp, x, d, passo_inicial, fator_reducao, η, count)

        # Atualização do ponto com a direção e o passo calculados
        x = x + α * d
        i += 1
        count.iteracoes = i
    end

    println("f(x) = ", obj(nlp, x))
    println("|grad_f(x)| = ", norm(d))
    # Atualizando as métricas
    count.function_value = obj(nlp, x)
    count.gradient_norm_value = norm(d)
end



#================================================================================================================
`BFGS(nlp, x, passo_inicial, fator_reducao, η, count, criterio_parada_grad, criterio_parada_i)`

# Objetivo
    Aplicar o método BFGS com busca de Armijo para calcular o mínimo da função.

# Saída
    - Print do valor da função e da norma do gradiente ao final.
    - Atualiza `count` com métricas de desempenho.

# Entradas
    - `nlp`                    : Problema de teste importado pela função `CUTEstModel()`.
    - `x`                      : Ponto inicial.
    - `passo_inicial`          : Passo inicial para a busca de Armijo.
    - `fator_reducao`          : Fator de redução aplicado na busca de Armijo.
    - `η`                      : Parâmetro da condição de Armijo.
    - `count`                  : Estrutura para acumular métricas.
    - `criterio_parada_grad`   : Critério de parada para a norma do gradiente.
    - `criterio_parada_i`      : Número máximo de iterações.

=================================================================================================================#
function BFGS(nlp, x, passo_inicial, fator_reducao, η, count, criterio_parada_grad, criterio_parada_i)
    i = 0
    H = I(nlp.meta.nvar)

    g = grad(nlp, x)
    count.gradient_count += 1
    d = -H * g
    ys_min = 0
    ys = 1

    while (norm(g) > criterio_parada_grad) && (i < criterio_parada_i)
        # Definindo o gradiente
        d = -H * g
        count.gradient_count += 1

        # Busca de Armijo
        α = armijo_search(nlp, x, d, passo_inicial, fator_reducao, η, count)

        # Atualização do ponto com a direção e o passo calculados
        x1 = x + α * d
        g1 = grad(nlp, x1)
        count.gradient_count += 1

        s = x1 - x
        y = g1 - g

        if !isnan(y' * s) && (y' * s) != 0.0
            ys = (y' * s)
            if ys != 0 && ys < ys_min
                ys_min = ys
            end
        end

        ρ = 1 / ys
        V = I - ρ * (s * y')
        H = V * H * V' + ρ * (s * s')

        x = x1
        g = g1
        i += 1
        count.iteracoes = i
    end

    println("f(x) = ", obj(nlp, x))
    println("|grad_f(x)| = ", norm(d))
    println("ys_min = ", ys_min)
    # Atualizando as métricas
    count.function_value = obj(nlp, x)
    count.gradient_norm_value = norm(d)
end


#================================================================================================================
`Spectral(nlp, x0, busca, passo_inicial, fator_reducao, η, count, criterio_parada_grad, criterio_parada_i)`

# Objetivo
    Aplicar o método Spectral com ou sem busca de Armijo para minimizar a função.

# Saída
    - Print do valor da função e da norma do gradiente ao final.
    - Atualiza `count` com métricas de desempenho.

# Entradas
    - `nlp`                    : Problema de teste importado pela função `CUTEstModel()`.
    - `x0`                     : Ponto inicial.
    - `busca`                  : Tipo de busca ("", "Armijo" ou "Armijo2").
    - `passo_inicial`          : Passo inicial para a busca de Armijo.
    - `fator_reducao`          : Fator de redução aplicado na busca de Armijo.
    - `η`                      : Parâmetro da condição de Armijo.
    - `count`                  : Estrutura para acumular métricas.
    - `criterio_parada_grad`   : Critério de parada para a norma do gradiente.
    - `criterio_parada_i`      : Número máximo de iterações.
=================================================================================================================#
function Spectral(nlp, x0, busca, passo_inicial, fator_reducao, η, count, criterio_parada_grad, criterio_parada_i)
    i = 0
    g = grad(nlp, x0)
    count.gradient_count += 1

    x1 = x0 - 0.01 * g
    g = grad(nlp, x1)
    count.gradient_count += 1
    hist_f = [obj(nlp, x1)]
    σ = 0.00001

    while (norm(g) > criterio_parada_grad) && (i < criterio_parada_i)
        δx = x1 - x0
        δy = grad(nlp, x1) - grad(nlp, x0)
        count.gradient_count += 2

        if ((δx' * δy) / (δx' * δx)) != 0 && !isnan((δx' * δy) / (δx' * δx)) || i == 0
            σ = (δx' * δy) / (δx' * δx)
        end

        g = grad(nlp, x1)
        d = -g * (1 / σ)
        count.gradient_count += 1

        # Busca de Armijo
        if busca == "Armijo"
            α = armijo_search(nlp, x1, d, passo_inicial, fator_reducao, η, count)
        elseif busca == "Armijo2"
            k = 0
            α = passo_inicial
            M = maximum(hist_f)
            count.function_count += 2
            while (obj(nlp, x1 + α * d) > M - η * α * d' * d) && k < 50
                k += 1
                count.function_count += 2
                α *= fator_reducao
            end
        else
            α = 1/σ
        end

        # Atualização do ponto com a direção e o passo calculados
        x0 = x1
        x1 = x0 + α * d

        push!(hist_f, obj(nlp, x1))
        if length(hist_f) > 10
            popfirst!(hist_f)
        end

        g = grad(nlp, x1)
        count.gradient_count += 1
        i += 1
        count.iteracoes = i
    end

    println("f(x) = ", obj(nlp, x1))
    println("|grad_f(x)| = ", norm(g))
    # Atualizando as métricas
    count.function_value = obj(nlp, x1)
    count.gradient_norm_value = norm(g)
end



tests = [
    "ARGLINA",
    "ARGLINB",
    "BA-L1SPLS",
    "BIGGS6",
    "BROWNAL",
    "COATING",
    "FLETCHCR",
    "GAUSS2LS",
    "GENROSE",
    "HAHN1LS",
    "HEART6LS",
    "HILBERTB",
    "HYDCAR6LS",
    "LANCZOS1LS",
    "LANCZOS2LS",
    "LRIJCNN1",
    "LUKSAN12LS",
    "LUKSAN16LS",
    "OSBORNEA",
    "PALMER1C",
    "PALMER3C",
    "PENALTY2",
    "PENALTY3",
    "QING",
    "ROSENBR",
    "STRTCHDV",
    "TESTQUAD",
    "THURBERLS",
    "TRIGON1",
    "TOINTGOR",
]

# Parâmetros gerais
passo_inicial = 1.0
fator_reducao = 0.5
parada_grad = 0.0001
parada_armijo = 0.0001
criterio_parada_i = 1000


for test in tests
    nlp = CUTEstModel(test; decode=true)
    println(test, " importado!")

    x0 = nlp.meta.x0

    # ------------------------- Cauchy -----------------------------

    count = metrics(0.0, 0.0, 0, 0, 0, 0.0)
    d = -grad(nlp, x0)
    count.gradient_count += 1
    println("Começando Cauchy com ", test)
    memoria = @allocated count.tempo = @elapsed Cauchy(nlp, x0, d, passo_inicial, fator_reducao, parada_armijo, count, parada_grad, criterio_parada_i)
    registro!(results_df, nlp, count, "Cauchy", count.iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, count.tempo, memoria)

    
    # ------------------------- BFGS -----------------------------
    count = metrics(0.0, 0.0, 0, 0, 0, 0.0)
    println("Começando BFGS com ", test)
    memoria = @allocated  count.tempo = @elapsed BFGS(nlp, x0, passo_inicial, fator_reducao, parada_armijo, count, parada_grad, criterio_parada_i)
    registro!(results_df, nlp, count, "BFGS", count.iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, count.tempo, memoria)

    # ------------------------- Spectral -----------------------------

    count = metrics(0.0, 0.0, 0, 0, 0, 0.0)
    println("Começando Spectral com ", test)
    memoria = @allocated count.tempo = @elapsed Spectral(nlp, x0, "", passo_inicial, fator_reducao, parada_armijo, count, parada_grad, criterio_parada_i)
    registro!(results_df, nlp, count, "Spectral", count.iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, count.tempo, memoria)

    # ------------------------- Spectral Armijo -----------------------------
    count = metrics(0.0, 0.0, 0, 0, 0, 0.0)
    println("Começando Spectral Armijo com ", test)
    memoria = @allocated count.tempo = @elapsed Spectral(nlp, x0, "Armijo", passo_inicial, fator_reducao, parada_armijo, count, parada_grad, criterio_parada_i)
    registro!(results_df, nlp, count, "Spectral_armijo", count.iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, count.tempo, memoria)

    # ------------------------- Spectral Armijo 2 -----------------------------
    count = metrics(0.0, 0.0, 0, 0, 0, 0.0)
    println("Começando Spectral Armijo 2 com ", test)
    memoria = @allocated count.tempo = @elapsed Spectral(nlp, x0, "Armijo2", passo_inicial, fator_reducao, parada_armijo, count, parada_grad, criterio_parada_i)
    registro!(results_df, nlp, count, "Spectral_armijo2", count.iteracoes, fator_reducao, passo_inicial, parada_grad, parada_armijo, count.tempo, memoria)


    finalize(nlp)
    println(test, " finalizado!")
end


println(results_df)