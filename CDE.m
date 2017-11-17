
function data_vector=CDE(data)
% Input: the data where each attribute value is represented by one
% unique integer
% Output: the new numerical data representatin
% Implemented by Songlei Jian
% Paper:17?IJCAI_Embedding-based Representation of Categorical Data by 
%       Hierarchical Value Coupling Learning

ini_time=cputime;
size(data);
[num_obj,num_att]=size(data);


num_value=0;
for i=1:num_att
    num_value_i=length(unique(data(:,i)));
    num_value=num_value+length(unique(data(:,i)));
end

%% build the value graph
g_value=zeros(num_value,num_value);
for i=1:num_obj
    for j=1:num_att
        for k=j:num_att
        index_j=data(i,j);
        index_k=data(i,k);
        if index_j==index_k
            g_value(index_j,index_k)=g_value(index_j,index_k)+1;
        else
            g_value(index_j,index_k)=g_value(index_j,index_k)+1;
            g_value(index_k,index_j)=g_value(index_k,index_j)+1;
        end
        end
    end
end

deg_table=diag(g_value); % which equals the |attribute_num|*frequency of each value
%% feature mutual information
nmi_value=ones(num_value,num_value);
for i=1:num_att-1
    for j=i:num_att
        nmi_ij=NMI(data(:,i)',data(:,j)');
        nmi_value(unique(data(:,i)),unique(data(:,j)))=nmi_ij;
        nmi_value(unique(data(:,j)),unique(data(:,i)))=nmi_ij;
    end
end

%% build the value influence matrix
fre_table=deg_table;
occurrence_matrix=zeros(num_value,num_value);
implicit_graph=zeros(num_value,num_value);
frequency_matrix=zeros(num_value,num_value);
mi_sim_matrix=zeros(num_value,num_value);
for i =1:num_value
    for j=1:num_value
        if i==j
            occurrence_matrix(i,j)=1;
            frequency_matrix(i,j)=1;
            mi_sim_matrix(i,j)=1;
        else
            if g_value(i,j)==0
                occurrence_matrix(i,j)=0;
                frequency_matrix(i,j)=fre_table(i)/fre_table(j);               
                mi_sim_matrix(i,j)=0;
            else
                occurrence_matrix(i,j)=g_value(i,j)/fre_table(i);
                mi_sim_matrix(i,j)=g_value(i,j)/(fre_table(i)*fre_table(j));  
                frequency_matrix(i,j)=nmi_value(i,j)*fre_table(i)/fre_table(j);
               
            end
        end
    end
end


%% d_o and d_r
dis_o=zeros(num_value,num_value);
dis_r=zeros(num_value,num_value);
for i=1:num_value
    for j=i:num_value
        D_o  = sqrt(sum((occurrence_matrix(i,:) - occurrence_matrix(j,:)) .^ 2));
        D_r  = sqrt(sum((frequency_matrix(i,:) - frequency_matrix(j,:)) .^ 2));
        dis_o(i,j)=D_o;
        dis_o(j,i)=D_o;
        dis_r(i,j)=D_r;
        dis_r(j,i)=D_r;
    end
end
dis_dif=abs(dis_o-dis_r);
a=sum(sum(dis_dif))/(num_value*num_value);
b=sum(sum(dis_r))/(num_value*num_value);
c=sum(sum(dis_o))/(num_value*num_value);
df=[a,b,c];

%% build the value-cluster space based on k-means
cluster_matrix=[];

flag=1;
i=2;
while flag
    cluster_result=kmeans(occurrence_matrix,i); 
    cluster_matrix_i=vec2matrix(cluster_result);
    [cluster_matrix_i_d,flag]=drop_cluster(cluster_matrix_i);
    cluster_matrix=horzcat(cluster_matrix,cluster_matrix_i_d);
    i=i+1;
        
end
k_cp=i;

flag=1;
i=2;
while flag    
    cluster_result_fre=kmeans(frequency_matrix,i);
    cluster_matrix_fre=vec2matrix(cluster_result_fre);
    [cluster_matrix_fre_d,flag]=drop_cluster(cluster_matrix_fre);
    cluster_matrix=horzcat(cluster_matrix,cluster_matrix_fre_d);
    i=i+1;         
end


dimension_before=size(cluster_matrix,2);


%% learning value embedding based on the value-cluster space
embedding_matrix=[];
[co,embedding_matrix,latent]=pca(cluster_matrix);

dimension=size(embedding_matrix,2);
i=1;
small_index=[];
b=10^(-10); % the parameter to control the dimension of value embedding

for i=1:dimension  
    if  (max(embedding_matrix(:,i))-min(embedding_matrix(:,i)))<b 
        small_index=[small_index,i];
    end
end
embedding_matrix(:,small_index)=[];
dimension_after=size(embedding_matrix,2);
time=cputime-ini_time;

%% calculate the object embedding 
data_vector=zeros(num_obj,num_att*size(embedding_matrix,2));
for i=1:num_obj
    row_data=[];
    for j=1:num_att
        value_vector=embedding_matrix(data(i,j),:);
        row_data=horzcat(row_data,value_vector); 
    end
    data_vector(i,:)=row_data; 
end



end


function l=update(g_value,i)
 neighbour=find(g_value(i,:)~=0);
 neighbour_i=setdiff(neighbour,i);
 max=0.5;
 l=0;
 for j=neighbour_i(1:length(neighbour_i))
     weight_ij=cal_weight(g_value,i,j);
      if weight_ij>max
         l=j;
         max=weight_ij;
     end
 end
end
function weight=cal_weight(g_value,i,j)
 neighbour=find(g_value(i,:)~=0);
 neighbour_i=setdiff(neighbour,i);
 neighbour_j=find(g_value(j,:)~=0);
 neighbour_k=intersect(neighbour_i,neighbour_j);
 % structure similarity
 struc_weight=0;
 a=1;
 b=1;
 for k=neighbour_k(1:length(neighbour_k))
     min_weight=min([g_value(i,j),g_value(i,k),g_value(k,j)]);
     max_weight=max([g_value(i,j),g_value(i,k),g_value(k,j)]);
     struc_weight=struc_weight+min_weight/max_weight;
 end
 struc_weight=struc_weight/(length(neighbour_j)-1);
 % connecting similarity
 con_weight=g_value(i,j)/g_value(j,j);
 weight=a*struc_weight+b*con_weight;
end
function cluster_matrix=vec2matrix(result)
row=length(result);
col=length(unique(result));
cluster_matrix=zeros(row,col);
uni=unique(result);
for i=1:col
    cluster_matrix(find(result==uni(i)),i)=1;    
    %cluster_matrix(find(result==uni(i)),i)=1-length(find(result==uni(i)))/row; 
end
end
function [cluster_matrix,flag]=drop_cluster(cluster_matrix)
[~,col]=size(cluster_matrix);
index=[];
flag=1;
a=10;
for i=1:col
    label_c=unique(cluster_matrix(:,i));

    if length(find(cluster_matrix(:,i)==label_c(1)))<=1 || length(find(cluster_matrix(:,i)==label_c(2)))<=1
        index=[index,i];
    end
end
cluster_matrix(:,index)=[];
if length(index)>ceil(col/a)
    flag=0;
end
end

