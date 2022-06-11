//
// Created by sun on 2022/1/1.
//

#include"layer.h"

Graph* mlp(int batch_size,int num_classes=10,const vector<int>& image_shape={1,28,28}){
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    data = graph->batch_flatten(data);
    auto fc1 = dense_add_bias(graph,data,128,"fc1");
    auto act1 = graph->relu(fc1);
    auto fc2 = dense_add_bias(graph,act1,64,"fc2");
    auto act2 = graph->relu(fc2);
    auto fc3 = dense_add_bias(graph,act2,num_classes,"fc3");
    auto net = graph->softmax(fc3);
    graph->function({net});
    return graph;
}