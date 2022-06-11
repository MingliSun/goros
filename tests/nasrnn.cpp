//
// Created by sun on 2022/1/19.
//
/*
 * Adapted from TASO
 */
#include"layer.h"
#define NASRNN_HIDDEN_SIZE 512
#define NASRNN_LENGTH 5
TensorHandle combine(Graph* graph,TensorHandle x,TensorHandle h){
    auto w1 = graph->new_weight({x->shape[1].get_upper_bound(),NASRNN_HIDDEN_SIZE});
    auto w2 = graph->new_weight({h->shape[1].get_upper_bound(),NASRNN_HIDDEN_SIZE});
    return graph->add(graph->dense(x,w1),graph->dense(h,w2));

}

TensorHandle nas_node(Graph* graph,TensorHandle inp,TensorHandle x){
    vector<TensorHandle> ts;
    ts.reserve(8);
    for(int i=0;i<8;i++){
        ts.push_back(combine(graph,x,inp));
    }
    vector<TensorHandle> midts;
    midts.push_back(graph->add(graph->relu(ts[0]),graph->sigmoid(ts[3])));
    midts.push_back(graph->add(graph->sigmoid(ts[1]),graph->tanh(ts[2])));
    midts.push_back(graph->add(graph->sigmoid(ts[4]),graph->tanh(ts[5])));
    midts.push_back(graph->add(graph->sigmoid(ts[6]),graph->relu(ts[7])));
    midts.push_back(graph->add(graph->sigmoid(midts[1]),graph->tanh(midts[2])));
    midts.push_back(graph->add(graph->tanh(midts[0]),graph->tanh(midts[3])));
    midts.push_back(graph->add(graph->tanh(midts[4]),graph->tanh(midts[5])));
    return graph->tanh(midts[6]);
}

Graph* nasrnn(){
    auto graph = new Graph;
    vector<TensorHandle> xs;
    xs.reserve(NASRNN_LENGTH);
    for(int i=0;i<NASRNN_LENGTH;i++){
        xs.push_back(graph->new_input({1,NASRNN_HIDDEN_SIZE}));
    }
    auto state = graph->new_weight({1,NASRNN_HIDDEN_SIZE});
    for(int i=0;i<NASRNN_LENGTH;i++){
        state = nas_node(graph,state,xs[i]);
    }
    return graph;
}

#undef NASRNN_HIDDEN_SIZE
#undef NASRNN_LENGTH