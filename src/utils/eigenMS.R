
user = "dmitrav"

# specify paths
data_path = paste0("/Users/", user, "/ETH/projects/normalization/data/")
save_to = paste0("/Users/", user, "/ETH/projects/normalization/res/other_methods/")
lib_source = paste0("/Users/", user, "/ETH/projects/normalization/src/utils/EigenMS/")

# upload scripts
source(paste0(lib_source, "EigenMS.R"))

# read and preprocess data
data = read.csv(paste0(data_path, "filtered_data.csv"))
info = data[,1:2]
data = log2(data[,-c(1:3)])  # transform
batch = read.csv(paste0(data_path, "batch_info.csv"))
batch = as.factor(batch$batch)

# apply EigenMS to normalize
m_ints_eig1 = eig_norm1(m=data, treatment=batch, prot.info=info)
m_ints_eig1$h.c = ceiling(0.2 * nrow(data))  # as recommended by authors for metabolomics data
m_ints_norm1 = eig_norm2(rv=m_ints_eig1)

# reverse transform
normalized = as.data.frame(t(2 ** m_ints_norm1$norm_m))

# save
write.csv(normalized, file=paste0(save_to, "eigenMS.csv"))
