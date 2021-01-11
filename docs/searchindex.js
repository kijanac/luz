Search.setIndex({docnames:["index","luz","luz.datasets","luz.events","luz.flags","luz.handlers","luz.learners","luz.modules","luz.optimizer","luz.predictors","luz.scorers","luz.trainers","luz.transforms","luz.tuners","luz.utils","modules"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,sphinx:56},filenames:["index.rst","luz.rst","luz.datasets.rst","luz.events.rst","luz.flags.rst","luz.handlers.rst","luz.learners.rst","luz.modules.rst","luz.optimizer.rst","luz.predictors.rst","luz.scorers.rst","luz.trainers.rst","luz.transforms.rst","luz.tuners.rst","luz.utils.rst","modules.rst"],objects:{"":{luz:[1,0,0,"-"]},"luz.datasets":{ConcatDataset:[2,1,1,""],Data:[2,1,1,""],Dataset:[2,1,1,""],OnDiskDataset:[2,1,1,""],Subset:[2,1,1,""],UnpackDataset:[2,1,1,""],WrapperDataset:[2,1,1,""],default_collate:[2,4,1,""],graph_collate:[2,4,1,""]},"luz.datasets.ConcatDataset":{cumulative_sizes:[2,2,1,""],datasets:[2,2,1,""]},"luz.datasets.Data":{keys:[2,3,1,""],to:[2,3,1,""]},"luz.datasets.Subset":{dataset:[2,2,1,""],indices:[2,2,1,""]},"luz.events":{Event:[3,1,1,""]},"luz.events.Event":{BATCH_ENDED:[3,2,1,""],BATCH_STARTED:[3,2,1,""],EPOCH_ENDED:[3,2,1,""],EPOCH_STARTED:[3,2,1,""],TESTING_ENDED:[3,2,1,""],TESTING_STARTED:[3,2,1,""],TRAINING_ENDED:[3,2,1,""],TRAINING_STARTED:[3,2,1,""]},"luz.flags":{Flag:[4,1,1,""]},"luz.flags.Flag":{TESTING:[4,2,1,""],TRAINING:[4,2,1,""]},"luz.handlers":{Handler:[5,1,1,""],Loss:[5,1,1,""],Progress:[5,1,1,""],Timer:[5,1,1,""]},"luz.handlers.Handler":{batch_ended:[5,3,1,""],batch_started:[5,3,1,""],epoch_ended:[5,3,1,""],epoch_started:[5,3,1,""],testing_ended:[5,3,1,""],testing_started:[5,3,1,""],training_ended:[5,3,1,""],training_started:[5,3,1,""]},"luz.handlers.Loss":{batch_ended:[5,3,1,""],epoch_started:[5,3,1,""]},"luz.handlers.Progress":{batch_ended:[5,3,1,""],epoch_started:[5,3,1,""],testing_ended:[5,3,1,""],testing_started:[5,3,1,""],training_ended:[5,3,1,""],training_started:[5,3,1,""]},"luz.handlers.Timer":{epoch_ended:[5,3,1,""],epoch_started:[5,3,1,""]},"luz.learners":{Learner:[6,1,1,""]},"luz.learners.Learner":{learn:[6,3,1,""]},"luz.modules":{AdditiveAttention:[7,1,1,""],AdditiveNodeAttention:[7,1,1,""],Concatenate:[7,1,1,""],Dense:[7,1,1,""],DenseRNN:[7,1,1,""],DotProductAttention:[7,1,1,""],EdgeAttention:[7,1,1,""],ElmanRNN:[7,1,1,""],GraphConv:[7,1,1,""],GraphConvAttention:[7,1,1,""],GraphNetwork:[7,1,1,""],MaskedSoftmax:[7,1,1,""],Module:[7,1,1,""],MultiheadEdgeAttention:[7,1,1,""],Reshape:[7,1,1,""],Squeeze:[7,1,1,""],Unsqueeze:[7,1,1,""]},"luz.modules.AdditiveAttention":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.AdditiveNodeAttention":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.Concatenate":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.Dense":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.DenseRNN":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.DotProductAttention":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.EdgeAttention":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.ElmanRNN":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.GraphConv":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.GraphConvAttention":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.GraphNetwork":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.MaskedSoftmax":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.Module":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.MultiheadEdgeAttention":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.Reshape":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.Squeeze":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.modules.Unsqueeze":{forward:[7,3,1,""],training:[7,2,1,""]},"luz.optimizer":{Optimizer:[8,1,1,""]},"luz.optimizer.Optimizer":{builder:[8,3,1,""],link:[8,3,1,""]},"luz.predictors":{Predictor:[9,1,1,""]},"luz.predictors.Predictor":{builder:[9,3,1,""],eval:[9,3,1,""],forward:[9,3,1,""],model:[9,2,1,""],predict:[9,3,1,""],to:[9,3,1,""],train:[9,3,1,""]},"luz.scorers":{CrossValidationScorer:[10,1,1,""],HoldoutValidationScorer:[10,1,1,""],Score:[10,1,1,""],Scorer:[10,1,1,""]},"luz.scorers.CrossValidationScorer":{score:[10,3,1,""]},"luz.scorers.HoldoutValidationScorer":{score:[10,3,1,""]},"luz.scorers.Score":{predictor:[10,2,1,""],score:[10,2,1,""]},"luz.scorers.Scorer":{score:[10,3,1,""]},"luz.trainers":{SupervisedTrainer:[11,1,1,""],Trainer:[11,1,1,""]},"luz.trainers.SupervisedTrainer":{run_batch:[11,3,1,""]},"luz.trainers.Trainer":{backward:[11,3,1,""],migrate:[11,3,1,""],optimizer_step:[11,3,1,""],process_batch:[11,3,1,""],run:[11,3,1,""],run_batch:[11,3,1,""],set_mode:[11,3,1,""]},"luz.transforms":{Argmax:[12,1,1,""],Compose:[12,1,1,""],DigraphToTensors:[12,1,1,""],Expand:[12,1,1,""],Identity:[12,1,1,""],Lookup:[12,1,1,""],NormalizePerTensor:[12,1,1,""],PowerSeries:[12,1,1,""],Transform:[12,1,1,""],Transpose:[12,1,1,""],ZeroMeanPerTensor:[12,1,1,""]},"luz.tuners":{BayesianTuner:[13,1,1,""],GridTuner:[13,1,1,""],RandomSearchTuner:[13,1,1,""],Tuner:[13,1,1,""]},"luz.tuners.BayesianTuner":{choose:[13,3,1,""],sample:[13,3,1,""]},"luz.tuners.GridTuner":{choose:[13,3,1,""],sample:[13,3,1,""]},"luz.tuners.RandomSearchTuner":{choose:[13,3,1,""],sample:[13,3,1,""]},"luz.tuners.Tuner":{best_hyperparameters:[13,3,1,""],best_score:[13,3,1,""],choose:[13,3,1,""],conditional:[13,3,1,""],get_sample:[13,3,1,""],hp_choose:[13,3,1,""],hp_sample:[13,3,1,""],pin:[13,3,1,""],sample:[13,3,1,""],score:[13,3,1,""],tune:[13,3,1,""]},"luz.utils":{adjacency:[14,4,1,""],batchwise_edge_mean:[14,4,1,""],batchwise_edge_sum:[14,4,1,""],batchwise_mask:[14,4,1,""],batchwise_node_mean:[14,4,1,""],batchwise_node_sum:[14,4,1,""],dot_product_attention:[14,4,1,""],expand_path:[14,4,1,""],in_degree:[14,4,1,""],int_length:[14,4,1,""],masked_softmax:[14,4,1,""],memoize:[14,4,1,""],mkdir_safe:[14,4,1,""],nodewise_edge_mean:[14,4,1,""],nodewise_edge_sum:[14,4,1,""],nodewise_mask:[14,4,1,""],out_degree:[14,4,1,""],set_seed:[14,4,1,""],temporary_seed:[14,4,1,""]},luz:{datasets:[2,0,0,"-"],events:[3,0,0,"-"],flags:[4,0,0,"-"],handlers:[5,0,0,"-"],learners:[6,0,0,"-"],modules:[7,0,0,"-"],optimizer:[8,0,0,"-"],predictors:[9,0,0,"-"],scorers:[10,0,0,"-"],trainers:[11,0,0,"-"],transforms:[12,0,0,"-"],tuners:[13,0,0,"-"],utils:[14,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:function"},terms:{"01261":7,"02907":7,"0473":7,"10903":7,"1409":7,"1609":7,"1710":7,"1806":7,"class":[2,3,4,5,6,7,8,9,10,11,12,13],"default":[6,7,10,11,14],"enum":[3,4],"float":[10,11],"function":[5,7,11],"int":[2,7,10,14],"new":10,"return":[2,5,6,7,8,9,10,11,13,14],"static":8,"true":[7,9,11],"while":7,Ege:14,abs:7,activ:7,addit:7,additiveattent:7,additivenodeattent:7,adjac:14,afterward:7,aggreg:[7,14],algorithm:[6,10,11],alia:10,all:[7,14],along:[7,14],although:7,ani:[13,14],arg:[6,7,8,9],argmax:12,argument:[7,14],arxiv:7,attent:[7,14],backward:11,bar_length:5,base:[2,3,4,5,6,7,8,9,10,11,12,13],batch:[2,3,7,11,14],batch_end:[3,5],batch_first:7,batch_start:[3,5],batchwis:14,batchwise_edge_mean:14,batchwise_edge_sum:14,batchwise_mask:14,batchwise_node_mean:14,batchwise_node_sum:14,bayesiantun:13,best_hyperparamet:13,best_scor:13,between:[7,14],bia:7,bidirect:7,bool:[7,11],both:7,builder:[8,9],call:7,callabl:[7,8,14],callback:5,care:7,choic:13,choos:13,classmethod:9,cls:6,collat:2,compos:12,compris:2,comput:[7,14],concatdataset:2,concaten:7,concen:7,condit:13,contain:[2,5],content:[0,15],convert:14,convolut:7,correspond:9,creat:[10,14],cross:10,crossvalidationscor:10,cumulative_s:2,custom:7,d_attn:7,d_e:[7,14],d_q:[7,14],d_u:7,d_v:7,data:[2,10,11],dataset:[1,6,10,11,13,15],default_col:2,defin:7,degre:[12,14],dens:7,densernn:7,desir:7,devic:[2,6,10,11,13],digraphtotensor:12,dim:[7,12,14],dimens:[7,14],dir:14,directori:[2,14],disk:2,domain:[2,9],dot:[7,14],dot_product_attent:14,dotproductattent:7,dropout:7,dure:[5,11],each:[7,14],edg:[7,14],edge_index:[7,14],edge_model:7,edgeattent:7,elmanrnn:7,els:11,end:3,enumer:[3,4],epoch:[3,5,11],epoch_end:[3,5],epoch_start:[3,5],equat:13,error:10,estim:10,eval:9,event:[1,15],everi:[5,7],expand:12,expand_path:14,fals:[12,13],featur:[7,14],feed:7,field:10,file:2,first:11,flag:[1,5,15],fold:10,fold_se:10,former:7,forward:[7,9],fraction:10,from:[2,7,9,11],func:14,gener:2,get:11,get_sampl:13,given:[6,7],global:7,global_model:7,graph:[2,7,14],graph_col:2,graphconv:7,graphconvattent:7,graphnetwork:7,gridtun:13,handler:[1,11,15],head:7,hidden_s:7,holdout:10,holdout_fract:10,holdoutvalidationscor:10,hook:7,hp_choos:13,hp_sampl:13,http:7,ident:12,if_fals:13,if_tru:13,ignor:[7,14],in_degre:14,incom:14,ind:5,index:[0,7,14],indic:[2,7,14],inherit:5,initi:7,input:7,input_s:7,instanc:[7,10],instead:7,int_length:14,intern:7,iter:[2,7,13],its:10,keepdim:12,kei:[2,7,14],kwarg:[2,5,6,7,8,9,13],label:[2,9],last:11,latter:7,layer:7,learn:[6,10],learner:[1,10,13,15],length:7,link:8,list:2,loader:5,loader_kwarg:11,lookup:12,lookup_dict:12,loss:[5,11],lower:13,mask:[7,14],masked_softmax:14,maskedsoftmax:7,matrix:14,memoiz:14,method:10,migrat:11,minim:11,mkdir_saf:14,mode:9,model:9,modul:[0,15],multihead:7,multiheadedgeattent:7,multipl:2,n_e:[7,14],n_v:[7,14],need:7,network:7,neural:7,node:[7,14],node_model:7,nodewis:[7,14],nodewise_edge_mean:14,nodewise_edge_sum:14,nodewise_mask:14,none:[2,5,6,7,9,10,11,12,14],nonlinear:7,normalizepertensor:12,notimplementederror:11,num_fold:10,num_head:7,num_iter:13,num_lay:7,number:[7,10],object:[2,5,6,8,9,10,11,12,13],ondiskdataset:2,one:7,optim:[1,11,15],optim_cl:8,optimizer_step:11,option:[6,7,10,11,14],org:7,out:[7,14],out_degre:14,out_shap:7,output:[7,11],output_s:7,overridden:7,packag:[0,15],page:0,pair:[7,14],paramet:[2,6,7,10,11,14],pass:7,path:14,perform:[5,7],pin:13,point:2,possibl:2,powerseri:12,predict:9,predictor:[1,6,8,10,11,15],print_interv:5,process:5,process_batch:11,product:[7,14],progress:5,properti:[2,13],protocol:6,pytorch:[7,9],queri:[7,14],rais:11,random:10,randomsearchtun:13,read:2,recip:7,regist:7,reshap:7,root:2,run:[7,11],run_batch:11,same:7,sampl:13,scale:[7,14],score:[6,10,13],scorer:[1,13,15],scriptmodul:7,search:0,seed:[10,14],seed_loop:13,sequenc:2,set:[9,10],set_mod:11,set_se:14,shape:[7,14],share:7,should:[5,7],silent:7,sinc:7,singl:11,softmax:[7,14],split:10,squeez:7,start:3,start_epoch:11,state:7,stop_epoch:11,store:2,str:[2,6,10,11,14],subclass:7,submodul:15,subset:2,sum:14,summari:7,supervisedtrain:11,t_co:2,take:[7,9],target:11,temporary_se:14,tensor:[7,9,11,14],tensortransform:12,test:[3,4,6,10,11],test_dataset:6,testing_end:[3,5],testing_start:[3,5],them:7,thi:7,timer:5,torch:[2,6,7,9,10,11,14],train:[3,4,5,6,7,9,11],trainer:[1,6,15],training_end:[3,5],training_start:[3,5],transform:[1,15],transpos:12,tune:13,tuner:[1,15],tupl:[10,11],type:[2,5,6,7,8,9,10,11,13,14],union:[6,10,11],unpackdataset:2,unsqueez:7,unsueez:7,updat:7,upper:13,use:[6,10,11],used:[6,9],using:[7,10,11,14],util:[1,2,15],val_dataset:[6,11],valid:[6,10,11],valu:[3,4,9],variou:5,vector:[7,14],weight:7,which:[2,5,9,10,14],within:7,wrapperdataset:2,zeromeanpertensor:12},titles:["Welcome to luz\u2019s documentation!","luz package","luz.datasets module","luz.events module","luz.flags module","luz.handlers module","luz.learners module","luz.modules module","luz.optimizer module","luz.predictors module","luz.scorers module","luz.trainers module","luz.transforms module","luz.tuners module","luz.utils module","luz"],titleterms:{content:1,dataset:2,document:0,event:3,flag:4,handler:5,indic:0,learner:6,luz:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],modul:[1,2,3,4,5,6,7,8,9,10,11,12,13,14],optim:8,packag:1,predictor:9,scorer:10,submodul:1,tabl:0,trainer:11,transform:12,tuner:13,util:14,welcom:0}})