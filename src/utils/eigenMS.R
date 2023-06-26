user <- "andreidm"

# specify paths
data_path <- paste0("/Users/", user, "/ETH/projects/normalization/data/sarah/")
save_to <- paste0("/Users/", user, "/ETH/projects/normalization/res/sarahs/other_methods/")
lib_source <- paste0("/Users/", user, "/ETH/projects/normalization/src/utils/EigenMS/")

# upload scripts
source(paste0(lib_source, "EigenMS.R"))

# read and preprocess data
data <- read.csv(paste0(data_path, "filtered_data.csv"))
info <- cbind(data[, 1], data[, 1])

# transform
data <- log2(data[, -1])
# replace -Inf with a minimal value found
data <- as.data.frame(lapply(data, function(x) replace(x, is.infinite(x), -15.497007)))

batch <- read.csv(paste0(data_path, "batch_info.csv"))
batch <- as.factor(batch$batch)

# apply EigenMS to normalize
m_ints_eig1 <- eig_norm1(m = data, treatment = batch, prot.info = info) # runs 109 minutes
m_ints_eig1$h.c <- ceiling(0.2 * nrow(data)) # as recommended by authors for metabolomics data

# work around since it complaints about non-unique rownames
rownames(m_ints_eig1$pres) <- paste(rownames(m_ints_eig1$pres), 1:nrow(m_ints_eig1$pres), sep = "_")

# apply normalization
m_ints_norm1 <- eig_norm2(rv = m_ints_eig1) # runs 2 minutes

# reverse transform
normalized <- as.data.frame(t(2**m_ints_norm1$norm_m))

# save
write.csv(normalized, file = paste0(save_to, "eigenMS.csv"))
