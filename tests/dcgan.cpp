//
// Created by sun on 2022/1/3.
//

/*
 * Net of the generator of DCGAN
 *   Reference:
 *  Radford, Alec, Luke Metz, and Soumith Chintala.
 *  "Unsupervised representation learning with deep convolutional generative adversarial networks."
 *   arXiv preprint arXiv:1511.06434 (2015).
 *
 */

#include"layer.h"
TensorHandle deconv2d(Graph* graph,TensorHandle input,int oc,int kernel,int stride,const string& name){
    //todo switch case
    return conv2d_transpose(graph,input,oc,kernel,stride,name,{1,0,1,0});
}

TensorHandle deconv2d_bn_relu(Graph* graph,TensorHandle input,int oc,int kernel,int stride,const string& name){
    float eps = 1e-5;
    auto net = deconv2d(graph,input,oc,kernel,stride,name+"_deconv");
    net = batch_norm_infer(graph,net,eps,name+"_bn");
    return graph->relu(net);
}

Graph* dcgan( int batch_size =1,
              int random_len=100,
              const vector<int>&  image_shape={3, 64, 64},
              int  ngf=128){
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto graph = new Graph;
    auto code = graph->new_input({batch_size,random_len},DT_FLOAT,"code");
    int k = code->shape.back().get_upper_bound();
    auto dense_weight = graph->new_weight({k,4*4*ngf*8},DT_FLOAT,"dense_weight");
    auto dense = graph->dense(code,dense_weight);
    auto relu = graph->relu(dense);
    relu = graph->reshape(relu,{-1,ngf*8,4,4});
    auto dc8 = deconv2d_bn_relu(graph,relu,512,4,2,"g2");
    auto dc16 = deconv2d_bn_relu(graph,dc8,256,4,2,"g3");
    auto dc32 = deconv2d_bn_relu(graph,dc16,128,4,2,"g4");
    auto dc64 = deconv2d_bn_relu(graph,dc32,3,4,2,"g5_deconv");
    auto tanh = graph->tanh(dc64);
    graph->function({tanh});
    return graph;
}

void test_dcgan(){
    auto graph = dcgan();
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_dcgan_1(){
    auto graph = new Graph;
    auto data = graph->new_input({1,32,4,4},DT_FLOAT,"data");
    auto dc32 = deconv2d_bn_relu(graph,data,128,4,2,"g4");
    auto dc64 = deconv2d_bn_relu(graph,dc32,3,4,2,"g5_deconv");
    graph->function({dc64});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

