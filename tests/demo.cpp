//
// Created by sun on 2021/11/18.
//

#include"layer.h"
#include<iostream>

void test_residual() {

    auto *graph = new Graph();
    TensorHandle x = graph->new_input({1, 3, 56, 56}, DT_FLOAT, "x");
    TensorHandle w1 = graph->new_weight({3, 3, 3, 3}, DT_FLOAT, "w1");
    TensorHandle w2 = graph->new_weight({3, 3, 1, 1}, DT_FLOAT, "w2");
    TensorHandle w3 = graph->new_weight({3, 3, 3, 3}, DT_FLOAT, "w3");
    TensorHandle left = graph->conv2d(x, w1, 1, 1, PD_MODE_SAME);
    left = graph->relu(left);
    TensorHandle left2 = graph->conv2d(left, w3, 1, 1, PD_MODE_SAME);
    TensorHandle right = graph->conv2d(x, w2, 1, 1, PD_MODE_SAME);
    right = graph->relu(right);
    TensorHandle res = graph->add(left2, right);
    graph->function({res});
    cout << graph->to_string() << endl;
    cout << "===========================" << endl;
    graph->optimize();
    cout << graph->to_string() << endl;
    graph->codegen_dot(string(__FUNCTION__).append(".dot"));
    graph->codegen_te(string(__FUNCTION__).append(".py"));

    graph->codegen_c(string(__FUNCTION__).append(".c"));
    graph->print_weights(string(__FUNCTION__).append(".txt"));
}

void test_fuse_bn2conv(){
    auto * graph = new Graph();

    TensorHandle x = graph->new_input({1,3,56,56},DT_FLOAT,"x");
    TensorHandle  w1 = graph->new_weight({3,3,3,3},DT_FLOAT,"w1");
    TensorHandle  mean = graph->new_weight({3l},DT_FLOAT,"mean");
    TensorHandle  var = graph->new_weight({3l},DT_FLOAT,"var");
    TensorHandle  scale = graph->new_weight({3l},DT_FLOAT,"scale");
    TensorHandle bias = graph->new_weight({3l},DT_FLOAT,"bias");
    TensorHandle one = graph->conv2d(x,w1,1,1,PD_MODE_SAME);
    TensorHandle res = graph->batch_norm(one, scale, bias, mean, var, 0.01);
    graph->function({res});
    cout << graph->to_string() << endl;
    cout << "===========================" << endl;
    graph->optimize();
    cout << graph->to_string() << endl;
    graph->codegen_dot(string(__FUNCTION__).append(".dot"));
    graph->codegen_te(string(__FUNCTION__).append(".py"));

    graph->codegen_c(string(__FUNCTION__).append(".c"));
    graph->print_weights(string(__FUNCTION__).append(".txt"));
}

void test_merge_2conv(){
    auto graph = new Graph();
    TensorHandle x = graph->new_input({1,3,56,56},DT_FLOAT,"x");
    TensorHandle act1 = graph->relu(x);
    TensorHandle  w1 = graph->new_weight({3,3,3,3},DT_FLOAT,"w1");
    TensorHandle  w2 = graph->new_weight({5,3,3,3},DT_FLOAT,"w2");
    TensorHandle conv1 = graph->conv2d(act1,w1,2,2,PD_MODE_SAME);
    TensorHandle conv2 = graph->conv2d(act1,w2,2,2,PD_MODE_SAME);

    graph->function({conv1,conv2});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void fuse_bn2group_conv(){
    const vector<int>& data_shape={1, 16, 224, 224};
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    auto weight = graph->new_weight({32,1,3,3},DT_FLOAT,"_conv1_weight");
    auto conv1 = graph->conv2d_group(data,weight,1,1,PD_MODE_SAME);
    auto bn1 = batch_norm_infer(graph,conv1,1e-4,"_bn1");
    graph->function({bn1});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_winograd(){
    auto graph = new Graph;
    TensorHandle x = graph->new_input({1,16,64,64},DT_FLOAT,"x");
    TensorHandle  w1 = graph->new_weight({16,16,3,3},DT_FLOAT,"w1");
//    TensorHandle y = graph->new_input({1,16,64,64},DT_FLOAT,"y");
//    TensorHandle  w2 = graph->new_weight({16,16,3,3},DT_FLOAT,"w2");
    TensorHandle conv1 = graph->conv2d(x,w1,1,1,PD_MODE_SAME);
//    auto conv2 = graph->conv2d(x,w2,1,1,PD_MODE_SAME);
//    auto net  = graph->concat({conv1,conv2},1);
//    graph->function({net});
    auto bn = batch_norm_infer(graph,conv1,2e-5,"bn");
    graph->function({bn});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_bn(){
    auto graph = new Graph;
    TensorHandle data = graph->new_input({1,16,64,64},DT_FLOAT,"data");
    TensorHandle data2 = graph->new_input({1,16,64,64},DT_FLOAT,"data2");
    auto conv1 = conv2d(graph,data,32,1,1,PD_MODE_SAME,"conv1");
    auto conv2 = conv2d(graph,data2,32,1,1,PD_MODE_SAME,"conv2");
    auto a = graph->add(conv1,conv2);
    auto net = batch_norm_infer(graph,a,2e-5,"bn");
    graph->function({net});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}
