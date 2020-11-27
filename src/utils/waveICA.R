
user = "dmitrav"

# specify paths
data_path = paste0("/Users/", user, "/ETH/projects/normalization/data/")
save_to = paste0("/Users/", user, "/ETH/projects/normalization/res/other_methods/")
lib_source = paste0("/Users/", user, "/ETH/projects/normalization/src/utils/waveICA/")

# upload scripts
source(paste0(lib_source, "WaveICA.R"))
source(paste0(lib_source, "normFact.R"))
source(paste0(lib_source, "unbiased_stICA.R"))
source(paste0(lib_source, "R2.R"))

# read and preprocess data
data = read.csv(paste0(data_path, "filtered_data.csv"))
data = as.data.frame(t(as.matrix(data[,-c(1:3)])))
batch = read.csv(paste0(data_path, "batch_info.csv"))
batch = batch$batch

# apply waveICA to normalize
data_wave_reconstruct = WaveICA(data=data, batch=batch)
normalized = as.data.frame(data_wave_reconstruct$data_wave)

# save
write.csv(normalized, file=paste0(save_to, "waveICA.csv"), sep=',')
