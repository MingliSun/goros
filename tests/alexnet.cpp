//
// Created by sun on 2022/1/19.
//

#include"layer.h"

Graph* alexnet(int num_classes=1000){
    auto graph = new Graph;
    auto data = graph->new_input({1,3,224,224},DT_FLOAT,"data");
    auto x = conv2d(graph,data,64,11,4,PD_MODE_SAME,"");
    x = graph->relu(x);
    x = graph->max_pool2d(x,3,3,2,2,PD_MODE_SAME);
    x = conv2d(graph,x,192,5,1,PD_MODE_SAME,"");
    x = graph->relu(x);
    x = graph->max_pool2d(x,3,3,2,2,PD_MODE_SAME);
    x = conv2d(graph,x,384,3,1,PD_MODE_SAME,"");
    x = graph->relu(x);
    x = conv2d(graph,x,256,3,1,PD_MODE_SAME,"");
    x = graph->relu(x);
    x = conv2d(graph,x,256,3,1,PD_MODE_SAME,"");
    x = graph->relu(x);
    x = graph->max_pool2d(x,3,3,2,2,PD_MODE_SAME);
    x = graph->batch_flatten(x);

    int k = x->shape.back().get_upper_bound();
    TensorHandle weight = graph->new_weight({k,4096},DT_FLOAT,"_weight");
    TensorHandle classifier = graph->dense(x,weight);
    classifier = graph->relu(classifier);
    classifier = graph->dropout(classifier,0.5);

    TensorHandle weight1 = graph->new_weight({4096,4096},DT_FLOAT,"_weight1");
    classifier = graph->dense(classifier,weight1);
    classifier = graph->relu(classifier);
    classifier = graph->dropout(classifier,0.5);

    TensorHandle weight2 = graph->new_weight({4096,num_classes},DT_FLOAT,"_weight2");
    classifier = graph->dense(classifier,weight2);
    graph->function({classifier});
    return graph;
}