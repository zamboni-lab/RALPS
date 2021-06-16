
user = "andreidm"

# specify paths (data 1, scenario 1)
data_path = paste0("/Users/", user, "/ETH/projects/normalization/data/sarah/")
save_to = paste0("/Users/", user, "/ETH/projects/normalization/res/sarahs/other_methods/")
lib_source = paste0("/Users/", user, "/ETH/projects/normalization/src/utils/waveICA/")

# upload scripts
source(paste0(lib_source, "WaveICA.R"))
source(paste0(lib_source, "normFact.R"))
source(paste0(lib_source, "unbiased_stICA.R"))
source(paste0(lib_source, "R2.R"))

# read and preprocess data
data = read.csv(paste0(data_path, "filtered_data.csv"))
metabolites = data[,1]
data = as.data.frame(t(as.matrix(data[,-1])))
batch_info = read.csv(paste0(data_path, "batch_info.csv"))
batch = batch_info$batch
group = batch_info$group

# apply waveICA to normalize without groups
data_wave_reconstruct = WaveICA(data=data, batch=batch)  # runs 3 minutes
normalized = as.data.frame(data_wave_reconstruct$data_wave)
colnames(normalized) = metabolites
# save
write.csv(normalized, file=paste0(save_to, "waveICA.csv"))


# apply waveICA to normalize with groups
group = batch_info$group
data_wave_reconstruct = WaveICA(data=data, batch=batch, group=group)  # runs 2 minutes
normalized = as.data.frame(data_wave_reconstruct$data_wave)
colnames(normalized) = metabolites
# save
write.csv(normalized, file=paste0(save_to, "waveICA_with_group.csv"))


