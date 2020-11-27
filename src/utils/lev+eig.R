
### LEV + EIG (from NOREVA 2.0)
### best normalization method for untargeted data with no QCs and no ISs:
### 1. Normalizing sample-wise with EigenMS,
### 2. Normalizing metabolite-wise with mean

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
data = data[,-c(1:3)]
samples_names = colnames(data)
data = log2(data)  # transform

# prepare sample-wise labels
sample_types = c()
for (i in 1:length(samples_names)) {
    type = paste(unlist(strsplit(samples_names[i], '_'))[1:3], collapse = '_')
    sample_types = c(sample_types, type)
}

unique_types = unique(sample_types)  # find unique sample types
sample_groups = rep(-1, length(sample_types))  # initialise sample groups
for (i in 1:length(unique_types)){
    # assign sample group
    sample_groups[sample_types == unique_types[i]] = i
}

sum(sample_groups < 0)  # check that all samples are assigned
sample_groups = as.factor(sample_groups)

# apply EigenMS to normalize sample-wise
m_ints_eig1 = eig_norm1(m=data, treatment=sample_groups, prot.info=info)
m_ints_eig1$h.c = ceiling(0.2 * nrow(data))  # as recommended by authors for metabolomics data
m_ints_norm1 = eig_norm2(rv=m_ints_eig1)

# reverse transform
normalized_step_1 = as.data.frame(t(2 ** m_ints_norm1$norm_m))
metabolite_means = apply(normalized_step_1, 2, mean)

# normalize metabolite-wise with mean
normalized_step_2 = apply(normalized_step_1, 1, function(x) x / metabolite_means)
normalized = as.data.frame(t(normalized_step_2))

# save
write.csv(normalized, file=paste0(save_to, "LEV+EIG.csv"))
