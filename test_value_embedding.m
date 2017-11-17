%% test file of CDE with clustering
dataName='soybeansmall';

load(dataName);
new_rep=CDE(data);

k=length(unique(label));
cluster_results=kmeans(new_rep,k);
