//
// Created by sun on 2021/12/27.
//
#include"Ops.h"
#include<algorithm>
// FIXME: batch flatten do not do any memory move(including load and store),so it is a PSEUDO operator,we should lower this to reshape
BatchFlatten::BatchFlatten(TensorHandle _input,Graph* g):OpBase(_input,OP_BATCH_FLATTEN,g) {
    assert(_input->shape.size()>2);
    int n = _input->shape.size();
    int size=1;
    for(int i=1;i<n;i++){
        size*= _input->shape[i].get_upper_bound();
    }
    vector<Affine> lambda;
    auto &batch = _input->shape[0];

    auto dimension = Graph::get_variable(graph->channel, size, 0, 1);
    Affine affine(dimension);
    for(int i=n-1;i>1;i--){
        Affine constant(_input->shape[i].get_upper_bound());
        lambda.push_back(affine.mod(constant));
        affine = affine.div(constant);
    }
    // 1 axis do not need mod
    lambda.emplace_back(affine);
    lambda.emplace_back(batch);//n
    reverse(lambda.begin(),lambda.end());
    auto t = new Tensor;
    graph->push(new Assign(t,_input,{batch,dimension},graph->num_cpt,lambda));
    outputs[0] = t;
}