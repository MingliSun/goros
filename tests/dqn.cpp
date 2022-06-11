//
// Created by sun on 2022/1/1.
//

/*
 * Net of Nature DQN
 * Reference:
 * Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
 * Nature 518.7540 (2015): 529.
 */

#include "layer.h"

Graph* dqn(int batch_size,int num_actions=18,const vector<int>& image_shape={4, 84, 84}){
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    auto conv1 = conv2d(graph,data,32,8,4,PD_MODE_SAME,"conv1");
    conv1 = bias_add(graph,conv1,"conv1");
    auto relu1 = graph->relu(conv1);

    auto conv2 = conv2d(graph,relu1,64,4,2,PD_MODE_SAME,"conv2");
    conv2 = bias_add(graph,conv2,"conv2");
    auto relu2 = graph->relu(conv2);

    auto conv3 = conv2d(graph,relu2,64,3,1,PD_MODE_SAME,"conv3");
    conv3 = bias_add(graph,conv3,"conv2");
    auto relu3 = graph->relu(conv3);

    auto flatten = graph->batch_flatten(relu3);
    auto dense1 = dense_add_bias(graph,flatten,512,"dense1");
    auto relu4 = graph->relu(dense1);
    auto dense2 = dense_add_bias(graph,relu4,num_actions,"dense2");
    graph->function({dense2});
    return graph;
}