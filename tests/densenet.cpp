//
// Created by sun on 2022/1/2.
//

#include "layer.h"

TensorHandle dense_layer(Graph *graph,TensorHandle input,int grown_rate,int bn_size,const string& name){
    auto bn1 = batch_norm_infer(graph,input,0.01, "batch_1_" + name);
    auto relu1 = graph->relu(bn1);
    auto conv1 = conv2d(graph,relu1,bn_size*grown_rate,1,1,PD_MODE_SAME, "conv2d_1_" + name);
    auto bn2 = batch_norm_infer(graph,conv1,0.01,"batch_2_"+name);
    auto relu2 = graph->relu(bn2);
    auto conv2 = conv2d(graph,relu2,grown_rate,3,1,PD_MODE_SAME,"conv2d_2_"+name);
    return conv2;
}

TensorHandle dense_block(Graph* graph,TensorHandle input,int num_layers,int bn_size,int growth_rate,const string& name){
    auto body = input;
    for(int i=0;i<num_layers;i++){
        body = dense_layer(graph,body,growth_rate,bn_size,name);
    }
    return body;
}

TensorHandle transition(Graph* graph,TensorHandle  input,int num_output_features,const string& name){
    auto bn = batch_norm_infer(graph,input,0.01,"batch_t_"+name);
    auto relu = graph->relu(bn);
    auto conv = conv2d(graph,relu,num_output_features,1,1,PD_MODE_SAME,"conv_t_"+name);
    return graph->avg_pool2d(conv,2,2,2,2,PD_MODE_SAME);
}

Graph* dense_net(int num_init_features,
                 int growth_rate,
                 const vector<int>& block_config,
                 const vector<int>& data_shape,
                 int bn_size=4,
                 int classes=1000){
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    auto conv1 = conv2d(graph,data,num_init_features,7,2,PD_MODE_SAME,"conv1");
    auto bn1 = batch_norm_infer(graph,conv1,0.01,"batch1");
    auto relu1 = graph->relu(bn1);
    auto mp = graph->max_pool2d(relu1,3,3,2,2,PD_MODE_SAME);
     int num_features = num_init_features;
     auto body = mp;
     for(uint i=0;i<block_config.size();i++){
         body = dense_block(graph,body,block_config[i],bn_size,growth_rate,::to_string(i));
         num_features += block_config[i]*growth_rate;
         if(i!=block_config.size()-1){
             body = transition(graph,body,num_features/2,::to_string(i));
             num_features = num_features/2;
         }
     }
     auto bn2 = batch_norm_infer(graph,body,0.01,"batch2");
     auto relu2 = graph->relu(bn2);
     auto avg = graph->avg_pool2d(relu2,7,7,1,1,PD_MODE_SAME);
     auto flat = graph->batch_flatten(avg);
     auto ret = dense_add_bias(graph,flat,classes,"dense");
     graph->function({ret});
     return graph;
}

void test_dense_net(int dense_net_size=121,
                    int classes=1000,
                    int batch_size=4,
                    const vector<int> &image_shape={3, 224, 224}){
    int num_init_features,growth_rate;
    vector<int> block_config;
    switch (dense_net_size) {
        case 121:
            num_init_features = 64;
            growth_rate = 32;
            block_config = {6,12,24,16};
            break;
        case 161:
            num_init_features = 96;
            growth_rate = 48;
            block_config = {6,12,36,24};
            break;
        case 169:
            num_init_features = 69;
            growth_rate = 32;
            block_config = {6,12,32,32};
            break;
        case 201:
            num_init_features = 64;
            growth_rate = 32;
            block_config = {6,12,48,322};
            break;
        default:
            throw exception();
    }
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto graph = dense_net(num_init_features, growth_rate, block_config, data_shape, batch_size, classes);

    string name = string(__FUNCTION__ )+::to_string(dense_net_size);
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(name+".dot");
    graph->codegen_te(name+".py");
}