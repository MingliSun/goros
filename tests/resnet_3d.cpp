//
// Created by sun on 2022/1/3.
//

/*
 * resnet ignoring batch_norm
 *  data height and weight is 224 (>32)
 */
#include"layer.h"

TensorHandle residual_unit_3d(Graph* graph, TensorHandle input, int oc, int stride, bool dim_match, const string& name, bool bottle_neck= true){
    if(bottle_neck){
        TensorHandle act1 = graph->relu(input);
        TensorHandle conv1 = conv3d(graph, act1, oc / 4, 1, stride, PD_MODE_SAME, name+"_conv1");
        TensorHandle act2 = graph->relu(conv1);
        TensorHandle conv2 = conv3d(graph, act2, oc / 4, 3, 1, PD_MODE_SAME, name+"_conv2");
        TensorHandle act3 = graph->relu(conv2);
        TensorHandle conv3 = conv3d(graph, act3, oc, 1, 1, PD_MODE_SAME, name+"_conv3");
        TensorHandle shortcut;
        if(dim_match) {
            shortcut = input;
        }else{
            shortcut = conv3d(graph,act1,oc,1,stride,PD_MODE_SAME,name+"_sc");
        }
        return  graph->element(conv3,shortcut,OP_EW_ADD);
    }

    TensorHandle act1 = graph->relu(input);
    TensorHandle conv1 = conv3d(graph, act1, oc , 3, stride, PD_MODE_SAME, name+"_conv1");
    TensorHandle act2 = graph->relu(conv1);
    TensorHandle conv2 = conv3d(graph, act2, oc , 3, 1, PD_MODE_SAME, name+"_conv2");
    TensorHandle shortcut;
    if(dim_match) {
        shortcut = input;
    }else{
        shortcut = conv3d(graph,act1,oc,1,stride,PD_MODE_SAME,name+"_sc");
    }
    return graph->element(conv2,shortcut,OP_EW_ADD);
}

Graph* resnet_3d(const vector<int>& units,int num_stages,const vector<int>& filter_list,int num_classes,const vector<int>& data_shape,bool bottle_neck=true){
    auto graph = new Graph();
    int num_units = units.size();
    assert(num_units==num_stages);
    TensorHandle data = graph->new_input(data_shape,DT_FLOAT,"data");

    TensorHandle body = conv3d(graph,data,filter_list[0],7,2,PD_MODE_SAME,"conv0");
    body = graph->relu(body);
    body = graph->max_pool2d(body, 3, 3, 2, 2, PD_MODE_SAME);
    for(int i=0;i<num_stages;i++){
        int stride;
        if(i==0) stride=1;
        else stride = 2;
        string name = "stage"+::to_string(i+1)+"_unit1";
        body = residual_unit_3d(graph, body, filter_list[i + 1], stride, false, name, bottle_neck);
        for(int j=0;j<units[i]-1;j++){
            string name1 = "stage"+::to_string(i+1)+"_unit"+::to_string(j+2);
            body = residual_unit_3d(graph, body, filter_list[i + 1], 1, true, name1, bottle_neck);
        }
    }
    TensorHandle relu1 = graph->relu(body);
    TensorHandle net = graph->global_avg_pool2d(relu1);
    net = graph->batch_flatten(net);
    net = dense_add_bias(graph,net,1000,"dense");
    net = graph->softmax(net);
    graph->function({net});
    return graph;
}
Graph* get_workload(int num_layers,int batch_size = 1,int num_classes = 1000,const vector<int>& image_shape = {3,16,112,112}){
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
    return resnet_3d(units,num_stages,filter_list,num_classes,data_shape,bottle_neck);

}
void test_resnet_3d(){
    int num_layers = 18;
    auto graph = get_workload(num_layers);
    string name = "resnet"+::to_string(num_layers);
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(name.append(".dot"));
    graph->codegen_te(name.append(".py"));
}