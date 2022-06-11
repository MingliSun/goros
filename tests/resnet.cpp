//
// Created by sun on 2021/12/24.
//

#include"Ops.h"
#include<iostream>

/*
 * resnet ignoring batch_norm
 *  data height and weight is 224 (>32)
 */
#include"layer.h"

TensorHandle resnet_block(Graph* graph, TensorHandle input, int oc, int stride, bool dim_match, const string& name, bool bottle_neck= true){
    if(bottle_neck){
        auto bn1 = batch_norm_infer(graph,input,2e-5,name+"_bn1");
        TensorHandle act1 = graph->relu(bn1);
        TensorHandle conv1 = conv2d(graph, act1, oc / 4, 1, stride, PD_MODE_SAME, name+"_conv1");
        auto bn2 = batch_norm_infer(graph,conv1,2e-5,name+"_bn2");
        TensorHandle act2 = graph->relu(bn2);
        TensorHandle conv2 = conv2d(graph, act2, oc / 4, 3, 1, PD_MODE_SAME, name+"_conv2");
        auto bn3 = batch_norm_infer(graph,conv2,2e-5,name+"_bn3");
        TensorHandle act3 = graph->relu(bn3);
        TensorHandle conv3 = conv2d(graph, act3, oc, 1, 1, PD_MODE_SAME, name+"_conv3");
        TensorHandle shortcut;
        if(dim_match) {
            shortcut = input;
        }else{
            shortcut = conv2d(graph,act1,oc,1,stride,PD_MODE_SAME,name+"_sc");
        }
        return  graph->add(conv3,shortcut);
    }
    auto bn1 = batch_norm_infer(graph,input,2e-5,name+"_bn1");
    TensorHandle act1 = graph->relu(bn1);
    TensorHandle conv1 = conv2d(graph, act1, oc , 3, stride, PD_MODE_SAME, name+"_conv1");
    auto bn2 = batch_norm_infer(graph,conv1,2e-5,name+"_bn2");
    TensorHandle act2 = graph->relu(bn2);
    TensorHandle conv2 = conv2d(graph, act2, oc , 3, 1, PD_MODE_SAME, name+"_conv2");
    TensorHandle shortcut;
    if(dim_match) {
        shortcut = input;
    }else{
        shortcut = conv2d(graph,act1,oc,1,stride,PD_MODE_SAME,name+"_sc");
    }
    return graph->element(conv2,shortcut,OP_EW_ADD);
}

Graph* resnet(const vector<int>& units,int num_stages,const vector<int>& filter_list,int num_classes,const vector<int>& data_shape,bool bottle_neck=true){
    auto graph = new Graph();
    int num_units = units.size();
    assert(num_units==num_stages);
    TensorHandle data = graph->new_input(data_shape,DT_FLOAT,"data");
    data = batch_norm_infer(graph,data,2e-5,"bn_data");
    TensorHandle body = conv2d(graph,data,filter_list[0],7,2,PD_MODE_SAME,"conv0");
    body = batch_norm_infer(graph,body,2e-5,"bn0");
    body = graph->relu(body);
    body = graph->max_pool2d(body, 3, 3, 2, 2, PD_MODE_SAME);
    for(int i=0;i<num_stages;i++){
        int stride;
        if(i==0) stride=1;
        else stride = 2;
        string name = "stage"+::to_string(i+1)+"_unit1";
        body = resnet_block(graph, body, filter_list[i + 1], stride, false, name, bottle_neck);
        for(int j=0;j<units[i]-1;j++){
            string name1 = "stage"+::to_string(i+1)+"_unit"+::to_string(j+2);
            body = resnet_block(graph, body, filter_list[i + 1], 1, true, name1, bottle_neck);
        }
    }
    auto bn1 = batch_norm_infer(graph,body,2e-5,"bn1");
    TensorHandle relu1 = graph->relu(bn1);
    TensorHandle net = graph->global_avg_pool2d(relu1);
    net = graph->batch_flatten(net);
    net = dense_add_bias(graph,net,1000,"dense");
    net = graph->softmax(net);
    graph->function({net});
    return graph;
}
Graph* get_resnet(int num_layers){
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
    return resnet(units,num_stages,filter_list,num_classes,data_shape,bottle_neck);

}
void test_resnet(){
    int num_layers = 50;
    auto graph = get_resnet(num_layers);
    string name = "resnet"+::to_string(num_layers);
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(name+".dot");
    graph->codegen_te(name+".py");
}

void test_resnet_1(){ //50
    auto graph = new Graph;
    TensorHandle data = graph->new_input({1,3,224,224},DT_FLOAT,"data");

    TensorHandle body = conv2d(graph,data,64,7,2,PD_MODE_SAME,"conv0");
    body = graph->relu(body);
    body = graph->max_pool2d(body, 3, 3, 2, 2, PD_MODE_SAME);
    body = resnet_block(graph, body, 256, 1, false, "block1", true);
    body = batch_norm_infer(graph,body,2e-5,"bn");
    graph->function({body});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}
void test_resnet_2(){//50
    auto graph = new Graph;
    //// test_resnet_1 result as input
    TensorHandle data = graph->new_input({1,256,56,56},DT_FLOAT,"data"); //assuming h and w are 128
    TensorHandle body = resnet_block(graph, data, 256, 1, true, "block2", true);
    body = resnet_block(graph, body, 256, 1, true, "block3", true);
    body = batch_norm_infer(graph,body,2e-5,"bn");
    graph->function({body});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_resnet_3(){
    auto graph = new Graph;
    auto data = graph->new_input({1,512,28,28},DT_FLOAT,"data");
    int stride = 2;
    string name = "stage3_unit1";
    auto body = resnet_block(graph, data, 1024, stride, false, name, true);

    graph->function({body});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_resnet_stage1(){//resnet50
    auto graph = new Graph;

    TensorHandle data = graph->new_input({1,3,224,224},DT_FLOAT,"data");
    TensorHandle body = conv2d(graph,data,64,7,2,PD_MODE_SAME,"conv0");
    body = graph->relu(body);
    body = graph->max_pool2d(body, 3, 3, 2, 2, PD_MODE_SAME);
    ////stage 1
    int stride = 1;
    string name = "stage1_unit1";
    body = resnet_block(graph, body, 256, stride, false, name, true);
    for(int j=0;j<2;j++){
        string name1 = "stage_unit"+::to_string(j+2);
        body = resnet_block(graph, body, 256, 1, true, name1, true);
    }
    graph->function({body});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_resnet_stage2(){
    auto graph = new Graph;
    //// stage1 as input
    auto data = graph->new_input({1,256,56,56},DT_FLOAT,"data");
    int stride = 2;
    string name = "stage2_unit1";
    auto body = resnet_block(graph, data, 512, stride, false, name, true);
    for(int j=0;j<3;j++){
        string name1 = "stage2_unit"+::to_string(j+2);
        body = resnet_block(graph, body, 512, 1, true, name1, true);
    }

    graph->function({body});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_resnet_stage3(){
    auto graph = new Graph;
    //// stage2 as input
    auto data = graph->new_input({1,512,28,28},DT_FLOAT,"data");
    int stride = 2;
    string name = "stage3_unit1";
    auto body = resnet_block(graph, data, 1024, stride, false, name, true);
    for(int j=0;j<5;j++){
        string name1 = "stage3_unit"+::to_string(j+2);
        body = resnet_block(graph, body, 1024, 1, true, name1, true);
    }

    graph->function({body});
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_resnet_stage4(){
    auto graph = new Graph;
    //// stage3 as input
    auto data = graph->new_input({1,1024,14,14},DT_FLOAT,"data");
    int stride = 2;
    string name = "stage4_unit1";
    auto body = resnet_block(graph, data, 2048, stride, false, name, true);
    for(int j=0;j<2;j++){
        string name1 = "stage4_unit"+::to_string(j+2);
        body = resnet_block(graph, body, 2048, 1, true, name1, true);
    }

    auto bn1 = batch_norm_infer(graph,body,2e-5,"bn1");
    TensorHandle relu1 = graph->relu(bn1);
    TensorHandle net = graph->global_avg_pool2d(relu1);
    net = graph->batch_flatten(net);
    net = dense_add_bias(graph,net,1000,"dense");
    net = graph->softmax(net);
    graph->function({net});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}