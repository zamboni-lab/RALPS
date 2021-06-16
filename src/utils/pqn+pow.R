
### PQN + POW (from NOREVA 2.0)
### second best normalization method for untargeted data with no QCs and no ISs:
### 1. Normalizing sample-wise with Probabilistic Quotient Normalization,
### 2. Normalizing metabolite-wise with intensity square root subtracts metabolite mean

user = "andreidm"

# specify paths
data_path = paste0("/Users/", user, "/ETH/projects/normalization/data/sarah/")
save_to = paste0("/Users/", user, "/ETH/projects/normalization/res/sarahs/other_methods/")
lib_source = paste0("/Users/", user, "/ETH/projects/normalization/src/utils/PQN/")

# upload scripts
source(paste0(lib_source, "PQN.R"))

# read and preprocess data
data = read.csv(paste0(data_path, "filtered_data.csv"))
metabolites = data[,1]
data = as.data.frame(t(as.matrix(data[,-1])))

# apply PQN 
normalized_step_1 = pqn(data, n="median")  # runs 6 minutes

# apply power transformation as in https://bmcgenomics.biomedcentral.com/articles/10.1186/1471-2164-7-142/tables/1
square_roots = sqrt(normalized_step_1)
mean_metabolite_roots = apply(square_roots, 2, mean)
normalized_step_2 = apply(square_roots, 1, function(x) x - mean_metabolite_roots)

normalized = as.data.frame(t(normalized_step_2))
colnames(normalized) = metabolites

# save
write.csv(normalized, file=paste0(save_to, "PQN+POW.csv"))
