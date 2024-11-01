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
end

# Criando DataFrame para armazenar os resultados
results_df = DataFrame(
    Problem = String[], 
    f_Cauchy = Union{Float64, Missing}[], grad_Cauchy = Union{Float64, Missing}[], 
    f_count_Cauchy = Union{Int, Missing}[], grad_count_Cauchy = Union{Int, Missing}[],
    f_BFGS = Union{Float64, Missing}[], grad_BFGS = Union{Float64, Missing}[], 
    f_count_BFGS = Union{Int, Missing}[], grad_count_BFGS = Union{Int, Missing}[],
    f_Spectral = Union{Float64, Missing}[], grad_Spectral = Union{Float64, Missing}[], 
    f_count_Spectral = Union{Int, Missing}[], grad_count_Spectral = Union{Int, Missing}[],
    f_Spectral2 = Union{Float64, Missing}[], grad_Spectral2 = Union{Float64, Missing}[], 
    f_count_Spectral2 = Union{Int, Missing}[], grad_count_Spectral2 = Union{Int, Missing}[],
    f_Spectral3 = Union{Float64, Missing}[], grad_Spectral3 = Union{Float64, Missing}[], 
    f_count_Spectral3 = Union{Int, Missing}[], grad_count_Spectral_armijo3 = Union{Int, Missing}[]
)


function registro!(results_df, nlp, count::metrics, algoritmo::String)
    # Obter o nome do problema atual
    problem_name = nlp.meta.name

    # Verificar se o DataFrame está vazio
    if isempty(results_df)
        # Se o DataFrame estiver vazio, cria uma nova linha diretamente
        push!(results_df, (
            Problem = problem_name,
            f_Cauchy = missing, grad_Cauchy = missing, f_count_Cauchy = missing, grad_count_Cauchy = missing,
            f_BFGS = missing, grad_BFGS = missing, f_count_BFGS = missing, grad_count_BFGS = missing,
            f_Spectral = missing, grad_Spectral = missing, f_count_Spectral = missing, grad_count_Spectral = missing,
            f_Spectral2 = missing, grad_Spectral2 = missing, f_count_Spectral2 = missing, grad_count_Spectral2 = missing,
            f_Spectral3 = missing, grad_Spectral3 = missing, f_count_Spectral3 = missing, grad_count_Spectral_armijo3 = missing
        ))
        # Definir `row_index` como a última linha do DataFrame
        row_index = nrow(results_df)
    else
        # Se o DataFrame não está vazio, buscar a linha correspondente
        row_index = findfirst(row -> row[:Problem] == problem_name, eachrow(results_df))

        # Se o problema não estiver no DataFrame, cria uma nova linha
        if isnothing(row_index)
            push!(results_df, (
                Problem = problem_name,
                f_Cauchy = missing, grad_Cauchy = missing, f_count_Cauchy = missing, grad_count_Cauchy = missing,
                f_BFGS = missing, grad_BFGS = missing, f_count_BFGS = missing, grad_count_BFGS = missing,
                f_Spectral = missing, grad_Spectral = missing, f_count_Spectral = missing, grad_count_Spectral = missing,
                f_Spectral2 = missing, grad_Spectral2 = missing, f_count_Spectral2 = missing, grad_count_Spectral2 = missing,
                f_Spectral3 = missing, grad_Spectral3 = missing, f_count_Spectral3 = missing, grad_count_Spectral_armijo3 = missing
            ))
            # Atualizar `row_index` para a última linha do DataFrame
            row_index = nrow(results_df)
        end
    end

    # Atualizar apenas as colunas específicas ao algoritmo atual
    if algoritmo == "Cauchy"
        results_df[row_index, [:f_Cauchy, :grad_Cauchy, :f_count_Cauchy, :grad_count_Cauchy]] = 
            (count.function_value, count.gradient_norm_value, count.function_count, count.gradient_count)
    elseif algoritmo == "BFGS"
        results_df[row_index, [:f_BFGS, :grad_BFGS, :f_count_BFGS, :grad_count_BFGS]] = 
            (count.function_value, count.gradient_norm_value, count.function_count, count.gradient_count)
    elseif algoritmo == "Spectral"
        results_df[row_index, [:f_Spectral, :grad_Spectral, :f_count_Spectral, :grad_count_Spectral]] = 
            (count.function_value, count.gradient_norm_value, count.function_count, count.gradient_count)
    elseif algoritmo == "Spectral_armijo"
        results_df[row_index, [:f_Spectral2, :grad_Spectral2, :f_count_Spectral2, :grad_count_Spectral2]] = 
            (count.function_value, count.gradient_norm_value, count.function_count, count.gradient_count)
    elseif algoritmo == "Spectral_armijo2"
        results_df[row_index, [:f_Spectral3, :grad_Spectral3, :f_count_Spectral3, :grad_count_Spectral_armijo3]] = 
            (count.function_value, count.gradient_norm_value, count.function_count, count.gradient_count)
    end
end


#================================================================================================================
# Função
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

function Spectral(nlp, x0, busca, passo_inicial, fator_reducao, count, criterio_parada_grad, criterio_parada_i)
    i=0

    g = grad(nlp,x0)
    count.gradient_count += 1

    x1 = x0 + 0.001*g

    g = grad(nlp,x1)
    count.gradient_count += 1

    hist_f = [obj(nlp, x1)]

    while (norm(g) > criterio_parada_grad) && (i < criterio_parada_i)
        
        δx = x1-x0
        δy = grad(nlp, x1)-grad(nlp, x0)
        count.gradient_count += 2

        println((δx' * δx))

        σ = (δx' * δy)/(δx' * δx)

        g=grad(nlp, x1)
        d=-g*(1/σ)
        count.gradient_count += 1
        
        α = passo_inicial
        c = fator_reducao

        if busca == "Armijo"
            k=0

            count.function_count += 2
            # Busca de Armijo
            while (obj(nlp, x1+α*d) > obj(nlp, x1)-c*α*d'*d) && k<50
                k+=1
                count.function_count += 2
                α = α*c
            end
        end

        if busca == "Armijo2"
            k=0

            M = maximum(hist_f)

            count.function_count += 2
            # Busca de Armijo
            while (obj(nlp, x1+α*d) > M-c*α*d'*d) && k<50
                k+=1
                count.function_count += 2
                α = α*c
            end
        end

        # Atualização do ponto com a direção e o passo calculados
        x0=x1
        x1=x0+α*d
    
        push!(hist_f, obj(nlp, x1))

        if length(hist_f) > 10
            popfirst!(hist_f)
        end

        g = grad(nlp,x1)

        i=i+1
        println("--------- Iteração: ",i," ---------")
        println("sigma = ",σ)
        println("f(x) = ",obj(nlp, x1))
        println("|grad_f(x)| = ", norm(g))
    
    end

    # Atualiando as métricas
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
    "HILBERTB"
]

for test in tests
    nlp = CUTEstModel(test; decode=true)
    println(test," importado!")

    # ------------------------- Spectral -----------------------------
    # Inicializar as métricas com zeros
    count = metrics(0,0,0,0)

    d = -grad(nlp,nlp.meta.x0)
    count.gradient_count = 1

    println("Começando Spectral com ", test)
    Spectral(nlp, nlp.meta.x0, "", 1, 0.5, count, 0.01, 1000)

    # Armazenando métricas no DataFrame
    println("Armazenando métricas")
    registro!(results_df,nlp,count,"Spectral")

    # ------------------------- Spectral Armijo -----------------------------
    # Inicializar as métricas com zeros
    count = metrics(0,0,0,0)

    d = -grad(nlp,nlp.meta.x0)
    count.gradient_count = 1

    println("Começando Spectral2 com ", test)
    Spectral(nlp, nlp.meta.x0, "Armijo", 1, 0.5, count, 0.01, 1000)

    # Armazenando métricas no DataFrame
    println("Armazenando métricas")
    registro!(results_df,nlp,count,"Spectral_armijo")

    # ------------------------- Spectral Armijo 2 -----------------------------

    # Inicializar as métricas com zeros
    count = metrics(0,0,0,0)

    d = -grad(nlp,nlp.meta.x0)
    count.gradient_count = 1

    println("Começando Spectral3 com ", test)
    Spectral(nlp, nlp.meta.x0, "Armijo2", 1, 0.5, count, 0.01, 1000)

    # Armazenando métricas no DataFrame
    println("Armazenando métricas")
    registro!(results_df,nlp,count,"Spectral_armijo2")

    finalize(nlp)
    println(test," finalizado!")
end


println(results_df)

Pkg.add("CSV")
using CSV

CSV.write("C:/Users/93dav/OneDrive/Documentos/Felipe/resultados_projeto_espectral/results_df.csv", results_df)