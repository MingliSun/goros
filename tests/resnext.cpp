//
// Created by sun on 2022/1/18.
//

#include"Ops.h"
#include<iostream>

/*
 * Adapted from https://github.com/vinx13/tvm-cuda-int8-benchmark/blob/master/model/resnext.py
 * ignore batch_norm
 */
#include"layer.h"
TensorHandle conv2d_group(Graph* graph, TensorHandle input, int oc, int kernel, int stride, PaddingMode p, int group,const string& name){
    int ic = input->shape[1].get_upper_bound();
    TensorHandle weight = graph->new_weight({oc,ic/group,kernel,kernel},DT_FLOAT,name+"_weight");
    return graph->conv2d_group(input,weight,stride,stride,p);
}

TensorHandle resnext_block(Graph* graph, TensorHandle input, int oc, int stride, bool dim_match,const string& name,bool bottle_neck=true,int num_group=32){
    if(bottle_neck){
        TensorHandle conv1 =conv2d_group(graph, input, oc / 2, 1, 1, PD_MODE_SAME, 1,name+"_conv1");
        TensorHandle act2 = graph->relu(conv1);
        TensorHandle conv2 = conv2d_group(graph, act2, oc / 2, 3, stride, PD_MODE_SAME, num_group,name+"_conv2");
        TensorHandle act3 = graph->relu(conv2);
        TensorHandle conv3 = conv2d(graph, act3, oc, 1, 1, PD_MODE_SAME, name+"_conv3");
        TensorHandle shortcut;
        if(dim_match) {
            shortcut = input;
        }else{
            shortcut = conv2d(graph,input,oc,1,stride,PD_MODE_SAME,name+"_sc");
        }
        auto eltwise =  graph->add(conv3,shortcut);
        return graph->relu(eltwise);
    }
    auto conv1 = conv2d(graph, input, oc, 3, stride, PD_MODE_SAME, name+"_conv1");
    auto act1 = graph->relu(conv1);
    auto conv2 = conv2d(graph,act1,oc,3,1,PD_MODE_SAME,"_conv2");
    TensorHandle shortcut;
    if(dim_match){
        shortcut = input;
    }else{
        shortcut = conv2d(graph,input,oc,1,stride,PD_MODE_SAME,"_sc");
    }
    auto eltwise = graph->add(conv2,shortcut);
    return graph->relu(eltwise);

}

Graph* resnext(const vector<int>& units,int num_stages,const vector<int>& filter_list,int num_classes,int num_group,const vector<int>& image_shape,bool bottle_neck=true){
    auto graph = new Graph();
    int num_units = units.size();
    assert(num_units==num_stages);
    TensorHandle data = graph->new_input(image_shape,DT_FLOAT,"data");

    TensorHandle body = conv2d(graph,data,filter_list[0],7,2,PD_MODE_SAME,"conv0");
    body = graph->relu(body);
    body = graph->max_pool2d(body, 3, 3, 2, 2, PD_MODE_SAME);
    for(int i=0;i<num_stages;i++){
        int stride;
        if(i==0) stride=1;
        else stride = 2;
        string name = "stage"+::to_string(i+1)+"_unit1";
        body = resnext_block(graph, body, filter_list[i + 1], stride, false,name,num_group);
        for(int j=0;j<units[i]-1;j++){
            string name1 = "stage"+::to_string(i+1)+"_unit"+::to_string(j+2);
            body = resnext_block(graph, body, filter_list[i + 1], 1, true, name1,num_group);
        }
    }
    TensorHandle net = graph->global_avg_pool2d(body);
    net = graph->batch_flatten(net);
    net = dense_add_bias(graph,net,num_classes,"dense");
    net = graph->softmax(net);
    graph->function({net});
    return graph;
}
Graph* get_resnext(int num_layers){
    int batch_size = 1;
    int num_classes = 1000;
    vector<int> image_shape = {3,224,224};
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    bool bottle_neck;
    vector<int> filter_list;
    if(num_layers>=50){
        filter_list = {64,256,512,1024,2048};
        bottle_neck  =true;
    }else{
        filter_list = {64,64,128,256,512};
        bottle_neck = false;
    }
    int num_stages = 4;
    vector<int> units;
    switch (num_layers) {
        case 18:
            units = {2,2,2,2};
            break;
        case 34:
        case 50:
            units = {3,4,6,3};
            break;
        case 101:
            units = {3,4,23,3};
            break;
        case 152:
            units = {3,8,36,3};
            break;
        case 200:
            units = {3,24,36,3};
            break;
        case 269:
            units = {3,30,48,8};
            break;
        default:
            throw exception();
//            throw "no experiments done on num_layers"+::to_string(num_layers);
    }
    return resnext(units,num_stages,filter_list,num_classes,32,data_shape,bottle_neck);

}
void test_resnext(){
    int num_layers = 50;
    auto graph = get_resnext(num_layers);
    string name = "resnext"+::to_string(num_layers);
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(name+".dot");
    graph->codegen_te(name+".py");
}