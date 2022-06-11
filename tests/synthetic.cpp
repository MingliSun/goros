//
// Created by sun on 2022/1/2.
//
/*
 * Synthetic networks for testing purposes. Ideally, these networks are similar in
 * structure to real world networks, but are much smaller in order to make testing
 * faster.
 */
#include"layer.h"

Graph* synthetic(const vector<int>& data_shape={1,3,24,12}){
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    data = graph->reshape(data,{-1,data_shape[3]});
    auto dense_weight = graph->new_weight({data_shape[3],data_shape[3]},DT_FLOAT,"dense_weight");
    auto dense = graph->dense(data,dense_weight);
    dense = graph->relu(dense);
    dense = graph->reshape(dense,{-1,data_shape[3]});
    auto conv = conv2d(graph,data,data_shape[1],3,1,PD_MODE_SAME,"conv");
    conv = graph->softmax(conv);
    auto added = graph->add(dense,conv);
    auto biased = bias_add(graph,added,"");
    biased = batch_norm_infer(graph,biased,0.01,"batch_norm");

    auto dense2 = graph->reshape(biased,{-1, data_shape[3]});
    auto dense2_weight = graph->new_weight({data_shape[3],data_shape[3]},DT_FLOAT,"dense2_weight");
    dense2 = graph->dense(dense2,dense2_weight);
    dense2 = graph->relu(dense2);
    dense2 = graph->reshape(dense2,data_shape);

    auto conv2 = conv2d(graph,biased,data_shape[1],3,1,PD_MODE_SAME,"conv2");
    conv2 = graph->softmax(conv2);
    auto added2 = graph->add(dense2,conv2);
    graph->function({added2});
    return graph;

}